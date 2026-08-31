/*
 * http_server.cpp
 * Socket handling and HTTP framing. Winsock on Windows, BSD sockets
 * elsewhere. The parser accepts only the routes it knows, caps the request
 * line, header block, and body length before allocating, and closes any
 * connection that violates those bounds.
 */
#include "ie/serving/http_server.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
#define IE_INVALID_SOCKET INVALID_SOCKET
#define IE_CLOSE_SOCKET closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
#define IE_INVALID_SOCKET (-1)
#define IE_CLOSE_SOCKET ::close
#endif

namespace ie {

namespace {

constexpr std::size_t kMaxHeaderBytes = 8 * 1024;
constexpr std::size_t kMaxBodyBytes = 1 * 1024 * 1024;

#ifdef _WIN32
struct WinsockGuard {
    WinsockGuard() {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
    }
    ~WinsockGuard() { WSACleanup(); }
};
#endif

std::string httpResponse(const std::string& status, const std::string& contentType,
                          const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

std::string errorResponse(const std::string& status, const std::string& message) {
    std::ostringstream oss;
    oss << "{\"error\":\"" << message << "\"}";
    return httpResponse(status, "application/json", oss.str());
}

void sendAll(SocketHandle socket, const std::string& payload) {
    std::size_t sent = 0;
    while (sent < payload.size()) {
        int written = ::send(socket, payload.data() + sent, static_cast<int>(payload.size() - sent), 0);
        if (written <= 0) return;
        sent += static_cast<std::size_t>(written);
    }
}

}

HttpServer::HttpServer(DynamicBatcher& batcher, std::string address, int port, std::size_t featureCount)
    : batcher_(batcher), address_(std::move(address)), port_(port), featureCount_(featureCount) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::stop() {
    running_ = false;
    if (listenSocket_ != -1) {
        IE_CLOSE_SOCKET(static_cast<SocketHandle>(listenSocket_));
        listenSocket_ = -1;
    }
}

std::string HttpServer::metricsJson() const {
    BatcherStats batchStats = batcher_.stats();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"requests\":" << stats_.requests.load() << ",\"errors\":" << stats_.errors.load()
        << ",\"batches\":" << batchStats.batches << ",\"batched_requests\":" << batchStats.requests
        << ",\"average_batch_size\":" << batchStats.averageBatchSize()
        << ",\"max_batch_observed\":" << batchStats.maxBatchObserved
        << ",\"total_batch_ms\":" << batchStats.totalBatchMs << "}";
    return oss.str();
}

std::string HttpServer::handleInfer(const std::string& body) {
    std::vector<float> features(featureCount_);

    if (body.size() == featureCount_ * sizeof(float)) {
        std::memcpy(features.data(), body.data(), body.size());
    } else if (body.size() == featureCount_) {
        for (std::size_t i = 0; i < featureCount_; ++i) {
            features[i] = static_cast<float>(static_cast<unsigned char>(body[i])) / 255.0f;
        }
    } else {
        ++stats_.errors;
        return errorResponse("400 Bad Request", "body must be " + std::to_string(featureCount_) +
                                                     " uint8 pixels or " +
                                                     std::to_string(featureCount_ * sizeof(float)) +
                                                     " float32 bytes");
    }

    for (float value : features) {
        if (!(value >= -1e6f && value <= 1e6f)) {
            ++stats_.errors;
            return errorResponse("400 Bad Request", "input contains non-finite or out-of-range values");
        }
    }

    try {
        std::future<std::vector<float>> future = batcher_.enqueue(std::move(features));
        std::vector<float> logits = future.get();

        std::size_t prediction =
            static_cast<std::size_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "{\"prediction\":" << prediction << ",\"logits\":[";
        for (std::size_t i = 0; i < logits.size(); ++i) {
            oss << logits[i];
            if (i + 1 < logits.size()) oss << ",";
        }
        oss << "]}";

        ++stats_.requests;
        return httpResponse("200 OK", "application/json", oss.str());
    } catch (const std::exception& ex) {
        ++stats_.errors;
        return errorResponse("500 Internal Server Error", ex.what());
    }
}

void HttpServer::handleClient(intptr_t clientHandle) {
    SocketHandle client = static_cast<SocketHandle>(clientHandle);
    std::string buffer;
    char chunk[4096];

    std::size_t headerEnd = std::string::npos;
    while (headerEnd == std::string::npos) {
        int received = ::recv(client, chunk, sizeof(chunk), 0);
        if (received <= 0) {
            IE_CLOSE_SOCKET(client);
            return;
        }
        buffer.append(chunk, static_cast<std::size_t>(received));
        headerEnd = buffer.find("\r\n\r\n");
        if (headerEnd == std::string::npos && buffer.size() > kMaxHeaderBytes) {
            sendAll(client, errorResponse("431 Request Header Fields Too Large", "headers too large"));
            IE_CLOSE_SOCKET(client);
            return;
        }
    }

    std::string headers = buffer.substr(0, headerEnd);
    std::string body = buffer.substr(headerEnd + 4);

    std::istringstream headerStream(headers);
    std::string requestLine;
    std::getline(headerStream, requestLine);
    if (!requestLine.empty() && requestLine.back() == '\r') requestLine.pop_back();

    std::istringstream requestParts(requestLine);
    std::string method, path;
    requestParts >> method >> path;

    std::size_t contentLength = 0;
    std::string headerLine;
    while (std::getline(headerStream, headerLine)) {
        if (!headerLine.empty() && headerLine.back() == '\r') headerLine.pop_back();
        std::size_t colon = headerLine.find(':');
        if (colon == std::string::npos) continue;
        std::string key = headerLine.substr(0, colon);
        std::transform(key.begin(), key.end(), key.begin(),
                        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (key == "content-length") {
            try {
                long long parsed = std::stoll(headerLine.substr(colon + 1));
                if (parsed < 0 || static_cast<std::size_t>(parsed) > kMaxBodyBytes) {
                    sendAll(client, errorResponse("413 Payload Too Large", "body too large"));
                    IE_CLOSE_SOCKET(client);
                    return;
                }
                contentLength = static_cast<std::size_t>(parsed);
            } catch (const std::exception&) {
                sendAll(client, errorResponse("400 Bad Request", "invalid content-length"));
                IE_CLOSE_SOCKET(client);
                return;
            }
        }
    }

    while (body.size() < contentLength) {
        int received = ::recv(client, chunk, sizeof(chunk), 0);
        if (received <= 0) break;
        body.append(chunk, static_cast<std::size_t>(received));
    }
    if (body.size() > contentLength) body.resize(contentLength);

    std::string response;
    if (method == "GET" && path == "/health") {
        response = httpResponse("200 OK", "application/json", "{\"status\":\"ok\"}");
    } else if (method == "GET" && path == "/metrics") {
        response = httpResponse("200 OK", "application/json", metricsJson());
    } else if (method == "POST" && path == "/infer") {
        response = handleInfer(body);
    } else {
        response = errorResponse("404 Not Found", "unknown route");
    }

    sendAll(client, response);
    IE_CLOSE_SOCKET(client);
}

void HttpServer::run() {
#ifdef _WIN32
    static WinsockGuard guard;
#endif

    SocketHandle server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server == IE_INVALID_SOCKET) {
        throw std::runtime_error("Failed to create server socket");
    }

    int reuse = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, address_.c_str(), &addr.sin_addr) != 1) {
        IE_CLOSE_SOCKET(server);
        throw std::runtime_error("Invalid bind address: " + address_);
    }

    if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        IE_CLOSE_SOCKET(server);
        throw std::runtime_error("Failed to bind " + address_ + ":" + std::to_string(port_));
    }
    if (::listen(server, 64) != 0) {
        IE_CLOSE_SOCKET(server);
        throw std::runtime_error("Failed to listen on socket");
    }

    listenSocket_ = static_cast<intptr_t>(server);
    running_ = true;

    std::cout << "Listening on http://" << address_ << ":" << port_ << "\n";
    std::cout << "  POST /infer    " << featureCount_ << " uint8 pixels or float32 values\n";
    std::cout << "  GET  /health\n";
    std::cout << "  GET  /metrics\n";

    while (running_) {
        SocketHandle client = ::accept(server, nullptr, nullptr);
        if (client == IE_INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }
        std::thread(&HttpServer::handleClient, this, static_cast<intptr_t>(client)).detach();
    }

    stop();
}

}
