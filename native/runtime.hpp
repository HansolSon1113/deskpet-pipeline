#pragma once
#include "deskpet_native.h"
#include "json.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using json = nlohmann::json;
inline void require(bool ok, const std::string &reason) { if (!ok) throw std::runtime_error(reason); }
inline json read_json(const std::string &path) {
    std::ifstream in(path); require(bool(in), "FILE_NOT_FOUND: " + path); return json::parse(in);
}
struct Job {
    json input; std::string output, error; std::atomic<bool> cancelled{false}; std::atomic<int> done{0};
};
class Model : public std::enable_shared_from_this<Model> {
    std::mutex mutex; std::condition_variable ready; std::deque<std::shared_ptr<Job>> pending;
    bool stopping = false; std::thread worker;
public:
    const json config;
    explicit Model(json c): config(std::move(c)) {}
    virtual ~Model() { stop(); }
    virtual json infer(Job &job) = 0;
    std::string path(const char *key) const { return config.at("paths").at(key).get<std::string>(); }
    void start() {
        worker = std::thread([this] {
            while (true) {
                std::shared_ptr<Job> job;
                { std::unique_lock<std::mutex> lock(mutex); ready.wait(lock,[&]{return stopping || !pending.empty();});
                  if (pending.empty() && stopping) return;
                  job = pending.front(); pending.pop_front(); }
                try { require(!job->cancelled, "CANCELLED"); job->output = infer(*job).dump();
                      require(!job->cancelled, "CANCELLED"); job->done.store(1, std::memory_order_release); }
                catch (const std::exception &e) {
                    job->error = e.what();
                    std::fprintf(stderr,"deskpet_native[%s]: %.2048s\n",config.value("role",std::string("unknown")).c_str(),e.what());
                    job->done.store(-1, std::memory_order_release);
                }
            }
        });
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; for (auto &j:pending) j->cancelled = true; }
        ready.notify_one(); if(worker.joinable()) worker.join();
    }
    std::shared_ptr<Job> submit(json input) {
        auto job = std::make_shared<Job>(); job->input = std::move(input);
        { std::lock_guard<std::mutex> lock(mutex); require(!stopping && pending.size()<16,"NATIVE_BUSY"); pending.push_back(job); }
        ready.notify_one(); return job;
    }
};
std::shared_ptr<Model> make_language_model(const json &config);
std::shared_ptr<Model> make_speech_model(const json &config);
std::vector<float> decode_pcm(const std::string &base64);
std::string encode_pcm(const float *samples, size_t count);
