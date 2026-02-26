/*
 * C7x Compute Service - RAII Utility Wrappers
 *
 * Move-only wrappers for file descriptors, mmap regions, and FILE*.
 * Zero-overhead in release builds (all methods inline).
 */

#ifndef RAII_H
#define RAII_H

#include <cstdio>
#include <cstddef>
#include <utility>
#include <unistd.h>
#include <sys/mman.h>

/* Owns a file descriptor; closes on destruction. */
class UniqueFd {
    int fd_ = -1;
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) : fd_(fd) {}
    ~UniqueFd() { if (fd_ >= 0) ::close(fd_); }

    UniqueFd(UniqueFd &&o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
    UniqueFd &operator=(UniqueFd &&o) noexcept {
        if (this != &o) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = std::exchange(o.fd_, -1);
        }
        return *this;
    }

    UniqueFd(const UniqueFd &) = delete;
    UniqueFd &operator=(const UniqueFd &) = delete;

    int get() const { return fd_; }
    int release() { return std::exchange(fd_, -1); }
    explicit operator bool() const { return fd_ >= 0; }
};

/* Owns an mmap'd region; munmaps on destruction. */
class MmapRegion {
    void *addr_ = nullptr;
    size_t len_ = 0;
public:
    MmapRegion() = default;
    MmapRegion(void *addr, size_t len) : addr_(addr), len_(len) {}
    ~MmapRegion() { if (addr_) ::munmap(addr_, len_); }

    MmapRegion(MmapRegion &&o) noexcept
        : addr_(std::exchange(o.addr_, nullptr)),
          len_(std::exchange(o.len_, 0)) {}
    MmapRegion &operator=(MmapRegion &&o) noexcept {
        if (this != &o) {
            if (addr_) ::munmap(addr_, len_);
            addr_ = std::exchange(o.addr_, nullptr);
            len_ = std::exchange(o.len_, 0);
        }
        return *this;
    }

    MmapRegion(const MmapRegion &) = delete;
    MmapRegion &operator=(const MmapRegion &) = delete;

    void *get() const { return addr_; }
    explicit operator bool() const { return addr_ != nullptr; }
};

/* Owns a FILE*; fcloses on destruction. */
class UniqueFile {
    FILE *fp_ = nullptr;
public:
    UniqueFile() = default;
    explicit UniqueFile(FILE *fp) : fp_(fp) {}
    ~UniqueFile() { if (fp_) std::fclose(fp_); }

    UniqueFile(UniqueFile &&o) noexcept : fp_(std::exchange(o.fp_, nullptr)) {}
    UniqueFile &operator=(UniqueFile &&o) noexcept {
        if (this != &o) {
            if (fp_) std::fclose(fp_);
            fp_ = std::exchange(o.fp_, nullptr);
        }
        return *this;
    }

    UniqueFile(const UniqueFile &) = delete;
    UniqueFile &operator=(const UniqueFile &) = delete;

    FILE *get() const { return fp_; }
    FILE *release() { return std::exchange(fp_, nullptr); }
    explicit operator bool() const { return fp_ != nullptr; }
};

#endif /* RAII_H */
