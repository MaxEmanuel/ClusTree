#include "http.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace tc {

namespace {

std::string urlDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit(static_cast<unsigned char>(s[i + 1])) &&
            isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else if (s[i] == '+') {
            out += ' ';
        } else {
            out += s[i];
        }
    }
    return out;
}

const char* statusText(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        default: return "Internal Server Error";
    }
}

const char* mimeType(const std::string& path) {
    auto ends = [&](const char* suf) {
        size_t n = std::strlen(suf);
        return path.size() >= n && path.compare(path.size() - n, n, suf) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".js")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json")) return "application/json";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".csv")) return "text/csv; charset=utf-8";
    return "application/octet-stream";
}

bool sendAll(int fd, const char* data, size_t len) {
    while (len > 0) {
        ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        data += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

void sendResponse(int fd, const HttpResponse& res) {
    std::ostringstream h;
    h << "HTTP/1.1 " << res.status << ' ' << statusText(res.status) << "\r\n"
      << "Content-Type: " << res.contentType << "\r\n"
      << "Content-Length: " << res.body.size() << "\r\n"
      << "Cache-Control: no-store\r\n"
      << "Connection: close\r\n\r\n";
    std::string head = h.str();
    if (sendAll(fd, head.data(), head.size())) sendAll(fd, res.body.data(), res.body.size());
}

}  // namespace

void HttpServer::route(const std::string& method, const std::string& path, Handler h) {
    routes_[method + " " + path] = std::move(h);
}

bool HttpServer::serveFile(const std::string& path, HttpResponse& res) const {
    if (staticDir_.empty() || path.find("..") != std::string::npos) return false;
    std::string file = staticDir_ + (path == "/" ? "/index.html" : path);
    std::ifstream in(file, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    res.body = ss.str();
    res.contentType = mimeType(file);
    return true;
}

void HttpServer::dispatch(const HttpRequest& req, HttpResponse& res) const {
    auto it = routes_.find(req.method + " " + req.path);
    if (it != routes_.end()) {
        try {
            it->second(req, res);
        } catch (const std::exception& e) {
            // Handlers signal invalid requests by throwing; report the message as JSON.
            std::string msg;
            for (char c : std::string(e.what())) {
                if (c == '"' || c == '\\') { msg += '\\'; msg += c; }
                else if (static_cast<unsigned char>(c) < 0x20) msg += ' ';
                else msg += c;
            }
            res = HttpResponse{};
            res.status = 400;
            res.body = "{\"error\":\"" + msg + "\"}";
        }
        return;
    }
    if (req.method == "GET" && serveFile(req.path, res)) return;
    res.status = 404;
    res.body = "{\"error\":\"not found\"}";
}

void HttpServer::handleConnection(int fd) const {
    timeval tv{60, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

    std::string buf;
    char chunk[65536];
    size_t headerEnd = std::string::npos;
    while (headerEnd == std::string::npos) {
        ssize_t n = ::recv(fd, chunk, sizeof chunk, 0);
        if (n <= 0 || buf.size() > (1u << 20)) { ::close(fd); return; }
        buf.append(chunk, static_cast<size_t>(n));
        headerEnd = buf.find("\r\n\r\n");
    }

    HttpRequest req;
    HttpResponse res;
    std::istringstream head(buf.substr(0, headerEnd));
    std::string target, version, line;
    head >> req.method >> target >> version;
    std::getline(head, line);
    while (std::getline(head, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = line.substr(0, colon), value = line.substr(colon + 1);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        value.erase(0, value.find_first_not_of(" \t"));
        req.headers[name] = value;
    }
    size_t qm = target.find('?');
    req.path = urlDecode(target.substr(0, qm));
    if (qm != std::string::npos) {
        std::istringstream qs(target.substr(qm + 1));
        std::string kv;
        while (std::getline(qs, kv, '&')) {
            size_t eq = kv.find('=');
            req.query[urlDecode(kv.substr(0, eq))] =
                eq == std::string::npos ? "" : urlDecode(kv.substr(eq + 1));
        }
    }

    size_t contentLength = 0;
    auto cl = req.headers.find("content-length");
    if (cl != req.headers.end()) contentLength = std::strtoull(cl->second.c_str(), nullptr, 10);
    if (contentLength > maxBodyBytes) {
        res.status = 413;
        res.body = "{\"error\":\"request body too large\"}";
        sendResponse(fd, res);
        ::close(fd);
        return;
    }
    req.body = buf.substr(headerEnd + 4);
    while (req.body.size() < contentLength) {
        ssize_t n = ::recv(fd, chunk, sizeof chunk, 0);
        if (n <= 0) { ::close(fd); return; }
        req.body.append(chunk, static_cast<size_t>(n));
    }

    dispatch(req, res);
    sendResponse(fd, res);
    ::shutdown(fd, SHUT_WR);
    ::close(fd);
}

bool HttpServer::listen(const std::string& host, int port) {
    std::signal(SIGPIPE, SIG_IGN);
    addrinfo hints{}, *ai = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &ai) != 0 || !ai) {
        std::fprintf(stderr, "cannot resolve %s\n", host.c_str());
        return false;
    }
    int srv = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (srv < 0 || ::bind(srv, ai->ai_addr, ai->ai_addrlen) != 0 || ::listen(srv, 64) != 0) {
        std::perror("bind/listen");
        freeaddrinfo(ai);
        if (srv >= 0) ::close(srv);
        return false;
    }
    freeaddrinfo(ai);
    for (;;) {
        int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([this, fd] { handleConnection(fd); }).detach();
    }
}

}  // namespace tc
