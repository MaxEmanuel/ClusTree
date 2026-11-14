// A deliberately small HTTP/1.1 server (POSIX sockets, one thread per connection,
// "Connection: close"). Sufficient for a single-user demonstration back end.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace tc {

struct HttpRequest {
    std::string method;
    std::string path;                           // without query string, URL-decoded
    std::map<std::string, std::string> query;   // decoded query parameters
    std::map<std::string, std::string> headers; // lower-cased names
    std::string body;

    std::string param(const std::string& key, const std::string& def = "") const {
        auto it = query.find(key);
        return it == query.end() ? def : it->second;
    }
};

struct HttpResponse {
    int status = 200;
    std::string contentType = "application/json";
    std::string body;
};

class HttpServer {
public:
    using Handler = std::function<void(const HttpRequest&, HttpResponse&)>;

    void route(const std::string& method, const std::string& path, Handler h);
    /// Files below dir are served for GET requests that match no route.
    void serveStatic(const std::string& dir) { staticDir_ = dir; }
    /// Blocks; returns false if the socket could not be set up.
    bool listen(const std::string& host, int port);

    size_t maxBodyBytes = 512u << 20;

private:
    std::map<std::string, Handler> routes_;  // "METHOD path"
    std::string staticDir_;

    void handleConnection(int fd) const;
    void dispatch(const HttpRequest& req, HttpResponse& res) const;
    bool serveFile(const std::string& path, HttpResponse& res) const;
};

}  // namespace tc
