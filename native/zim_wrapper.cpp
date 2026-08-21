// Thin C interface over the libzim C++ API, consumed from Python via ctypes.
#include <zim/archive.h>
#include <zim/entry.h>
#include <zim/item.h>
#include <zim/blob.h>
#include <zim/search.h>

#include <string>
#include <vector>
#include <cstring>
#include <new>

namespace {

struct Handle {
  zim::Archive* archive;
};

struct ResultSet {
  std::vector<std::string> titles;
  std::vector<std::string> paths;
  std::vector<int> scores;
  std::vector<std::string> snippets;
};

char* dup(const std::string& s) {
  char* p = new char[s.size() + 1];
  std::memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

void set_err(char* errbuf, int errlen, const std::string& msg) {
  if (errbuf && errlen > 0) {
    std::string m = msg;
    if (static_cast<int>(m.size()) >= errlen) {
      m = m.substr(0, errlen - 1);
    }
    std::memcpy(errbuf, m.c_str(), m.size() + 1);
  }
}

// Fill article out-params from an entry (following redirects). Returns 1 on success.
int fill_article(const zim::Entry& entry,
                 char** out_title, char** out_path, char** out_mimetype,
                 char** out_content, long* out_content_len) {
  zim::Item item = entry.getItem(true);
  if (out_title) *out_title = dup(item.getTitle());
  if (out_path) *out_path = dup(item.getPath());
  if (out_mimetype) *out_mimetype = dup(item.getMimetype());
  zim::Blob blob = item.getData();
  if (out_content) {
    *out_content = new char[blob.size() ? blob.size() : 1];
    if (blob.size()) std::memcpy(*out_content, blob.data(), blob.size());
  }
  if (out_content_len) *out_content_len = static_cast<long>(blob.size());
  return 1;
}

}  // namespace

extern "C" {

void* zimw_open(const char* path, char* errbuf, int errlen) {
  try {
    auto* h = new Handle();
    h->archive = new zim::Archive(std::string(path));
    return h;
  } catch (const std::exception& e) {
    set_err(errbuf, errlen, e.what());
    return nullptr;
  } catch (...) {
    set_err(errbuf, errlen, "unknown error opening zim");
    return nullptr;
  }
}

void zimw_close(void* h) {
  if (!h) return;
  auto* hh = static_cast<Handle*>(h);
  delete hh->archive;
  delete hh;
}

long zimw_article_count(void* h) {
  if (!h) return 0;
  return static_cast<long>(static_cast<Handle*>(h)->archive->getArticleCount());
}

long zimw_entry_count(void* h) {
  if (!h) return 0;
  return static_cast<long>(static_cast<Handle*>(h)->archive->getAllEntryCount());
}

int zimw_has_fulltext_index(void* h) {
  if (!h) return 0;
  return static_cast<int>(static_cast<Handle*>(h)->archive->hasFulltextIndex());
}

char* zimw_get_metadata(void* h, const char* name, int* ok) {
  if (ok) *ok = 0;
  if (!h) return nullptr;
  try {
    std::string v = static_cast<Handle*>(h)->archive->getMetadata(std::string(name));
    if (ok) *ok = 1;
    return dup(v);
  } catch (...) {
    if (ok) *ok = 0;
    return nullptr;
  }
}

int zimw_get_article_by_path(void* h, const char* path,
                             char** out_title, char** out_path, char** out_mimetype,
                             char** out_content, long* out_content_len) {
  if (out_title) *out_title = nullptr;
  if (out_path) *out_path = nullptr;
  if (out_mimetype) *out_mimetype = nullptr;
  if (out_content) *out_content = nullptr;
  if (out_content_len) *out_content_len = 0;
  if (!h) return 0;
  try {
    zim::Entry entry = static_cast<Handle*>(h)->archive->getEntryByPath(std::string(path));
    return fill_article(entry, out_title, out_path, out_mimetype, out_content, out_content_len);
  } catch (...) {
    return 0;
  }
}

int zimw_get_article_by_title(void* h, const char* title,
                              char** out_title, char** out_path, char** out_mimetype,
                              char** out_content, long* out_content_len) {
  if (out_title) *out_title = nullptr;
  if (out_path) *out_path = nullptr;
  if (out_mimetype) *out_mimetype = nullptr;
  if (out_content) *out_content = nullptr;
  if (out_content_len) *out_content_len = 0;
  if (!h) return 0;
  try {
    zim::Entry entry = static_cast<Handle*>(h)->archive->getEntryByTitle(std::string(title));
    return fill_article(entry, out_title, out_path, out_mimetype, out_content, out_content_len);
  } catch (...) {
    return 0;
  }
}

void* zimw_search(void* h, const char* query, int max_results,
                  char* errbuf, int errlen) {
  if (!h) return nullptr;
  try {
    auto* rs = new ResultSet();
    zim::Searcher searcher(*static_cast<Handle*>(h)->archive);
    zim::Query q;
    q.setQuery(std::string(query));
    zim::Search search = searcher.search(q);
    auto results = search.getResults(0, max_results);
    for (auto it = results.begin(); it != results.end(); ++it) {
      rs->titles.push_back(it.getTitle());
      rs->paths.push_back(it.getPath());
      rs->scores.push_back(it.getScore());
      rs->snippets.push_back(it.getSnippet());
    }
    return rs;
  } catch (const std::exception& e) {
    set_err(errbuf, errlen, e.what());
    return nullptr;
  } catch (...) {
    set_err(errbuf, errlen, "unknown search error");
    return nullptr;
  }
}

int zimw_search_count(void* rs) {
  if (!rs) return 0;
  return static_cast<int>(static_cast<ResultSet*>(rs)->titles.size());
}

int zimw_search_result(void* rs, int i,
                       char** title, char** path, int* score, char** snippet) {
  if (!rs) return 0;
  auto* r = static_cast<ResultSet*>(rs);
  if (i < 0 || i >= static_cast<int>(r->titles.size())) return 0;
  if (title) *title = dup(r->titles[i]);
  if (path) *path = dup(r->paths[i]);
  if (score) *score = r->scores[i];
  if (snippet) *snippet = dup(r->snippets[i]);
  return 1;
}

void zimw_search_free(void* rs) {
  if (rs) delete static_cast<ResultSet*>(rs);
}

void zimw_free(void* p) {
  if (p) delete[] static_cast<char*>(p);
}

}  // extern "C"
