// generate_mcst_games.cpp  --  C++ MCTS self-play data generator
//
// Build (pick one):
//   MSVC : cl /O2 /std:c++17 /EHsc generate_mcst_games.cpp
//   g++  : g++ -O3 -std=c++17 -pthread -o generate_mcst_games generate_mcst_games.cpp
//   clang: clang++ -O3 -std=c++17 -pthread -o generate_mcst_games generate_mcst_games.cpp
//
// Usage:
//   generate_mcst_games -b 9 -r 1000 -n 200 -w 4 -e oneplane -o data/mcts
//   generate_mcst_games -b 9 -r 1000 -n 200 -w 4 -e oneplane,sevenplane,elevenplane -o data/mcts
//   generate_mcst_games -b 9 -r 1000 -n 1000 -w 4 --checkpoint-every 100 -o data/mcts
//   generate_mcst_games -b 9 -r 800 -n 10000 -w 12 -e alphazero -o data/bootstrap
//
// Multi-encoder: each game is played once; all requested encoders are applied
// per position. Feature files are saved per encoder; labels are shared.
//
// Checkpoints: after every --checkpoint-every games a binary checkpoint is
// written so a crashed run can resume from where it left off.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <sstream>
#include <atomic>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <process.h>
#else
#include <thread>
#include <mutex>
#endif

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#pragma warning(disable : 4996) // fopen
#endif

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------
static inline int popcount64(uint64_t x) {
#ifdef _MSC_VER
#ifdef _M_X64
    return (int)__popcnt64(x);
#else
    return (int)__popcnt((uint32_t)x) + (int)__popcnt((uint32_t)(x >> 32));
#endif
#else
    return __builtin_popcountll(x);
#endif
}

static inline int ctz64(uint64_t x) {
#ifdef _MSC_VER
    unsigned long idx;
#ifdef _M_X64
    _BitScanForward64(&idx, x);
#else
    if ((uint32_t)x) _BitScanForward(&idx, (uint32_t)x);
    else { _BitScanForward(&idx, (uint32_t)(x >> 32)); idx += 32; }
#endif
    return (int)idx;
#else
    return __builtin_ctzll(x);
#endif
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int MAX_BOARD = 19;
static constexpr int MAX_N     = MAX_BOARD * MAX_BOARD; // 361
static constexpr int PW        = (MAX_N + 63) / 64;    // 6 words

enum { EMPTY = 0, BLACK = 1, WHITE = 2 };
static int other_color(int c) { return 3 - c; }

// ---------------------------------------------------------------------------
// PointSet  -- fixed-width bitset for up to MAX_N points
// ---------------------------------------------------------------------------
struct PointSet {
    uint64_t w[PW]{};

    void set(int i)       { w[i >> 6] |=  (1ULL << (i & 63)); }
    void clr(int i)       { w[i >> 6] &= ~(1ULL << (i & 63)); }
    bool test(int i) const{ return (w[i >> 6] >> (i & 63)) & 1; }

    bool empty() const {
        for (int i = 0; i < PW; i++) if (w[i]) return false;
        return true;
    }
    int count() const {
        int c = 0;
        for (int i = 0; i < PW; i++) c += popcount64(w[i]);
        return c;
    }
    PointSet operator|(const PointSet& o) const {
        PointSet r;
        for (int i = 0; i < PW; i++) r.w[i] = w[i] | o.w[i];
        return r;
    }
    PointSet operator-(const PointSet& o) const {
        PointSet r;
        for (int i = 0; i < PW; i++) r.w[i] = w[i] & ~o.w[i];
        return r;
    }
    void operator|=(const PointSet& o) {
        for (int i = 0; i < PW; i++) w[i] |= o.w[i];
    }
    template<typename F> void for_each(F fn) const {
        for (int i = 0; i < PW; i++) {
            uint64_t x = w[i];
            while (x) { int b = ctz64(x); fn(i * 64 + b); x &= x - 1; }
        }
    }
};

// ---------------------------------------------------------------------------
// PrevStates -- fixed-size POD array replacing std::vector<uint64_t>
// Eliminates heap allocation on every rollout and tree-node copy.
// 1024 entries covers any game up to ~512 moves (rollout start + rollout length).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BoardHistory -- circular buffer of the last 8 board states for AlphaZero encoder.
// ---------------------------------------------------------------------------
struct BoardHistory {
    static constexpr int DEPTH = 8;
    int8_t boards[DEPTH][MAX_N];
    int count = 0;

    void push(const int8_t* color_at, int N) {
        int shift = (count < DEPTH) ? count : DEPTH - 1;
        for (int i = shift; i > 0; i--)
            memcpy(boards[i], boards[i-1], N);
        memcpy(boards[0], color_at, N);
        if (count < DEPTH) count++;
    }
    const int8_t* get(int age) const {
        return (age < count) ? boards[age] : nullptr;
    }
};

struct PrevStates {
    // 2048 covers: game-tree depth (≤ max_moves) + rollout length (≤ board_size²)
    // for any board size up to 19×19 with max_moves ≤ 1600.
    static constexpr int MAX = 2048;
    uint64_t data[MAX];
    int      cnt = 0;
    void push_back(uint64_t v) {
        assert(cnt < MAX && "PrevStates overflow: increase MAX or cap rollout length");
        data[cnt++] = v;
    }
};

// ---------------------------------------------------------------------------
// Neighbor table + Zobrist hashes  (shared, immutable per board size)
// ---------------------------------------------------------------------------
struct NeighborInfo { int nb[4]; int cnt; };

struct BoardConfig {
    int rows, cols, N;
    NeighborInfo neighbors[MAX_N];
    uint64_t zobrist[MAX_N][2]; // [point][BLACK-1 or WHITE-1]
    uint64_t empty_hash;

    void init(int r, int c) {
        rows = r; cols = c; N = r * c;
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++) {
                int idx = row * cols + col, cnt = 0;
                if (row > 0)        neighbors[idx].nb[cnt++] = (row-1)*cols + col;
                if (row < rows - 1) neighbors[idx].nb[cnt++] = (row+1)*cols + col;
                if (col > 0)        neighbors[idx].nb[cnt++] = row*cols + (col-1);
                if (col < cols - 1) neighbors[idx].nb[cnt++] = row*cols + (col+1);
                neighbors[idx].cnt = cnt;
            }
        std::mt19937_64 rng(42);
        for (int i = 0; i < N; i++) { zobrist[i][0] = rng(); zobrist[i][1] = rng(); }
        empty_hash = rng();
    }
};

// ---------------------------------------------------------------------------
// GoString
// ---------------------------------------------------------------------------
struct GoString {
    int8_t color; // BLACK / WHITE / 0 = dead
    PointSet stones;
    PointSet liberties;
};

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------
class Board {
public:
    const BoardConfig* cfg = nullptr;
    int8_t  color_at[MAX_N]{};
    int16_t str_id  [MAX_N];
    std::vector<GoString> strs;
    uint64_t zhash = 0;

    void init(const BoardConfig* c) {
        cfg = c;
        memset(color_at, 0, sizeof(color_at));
        memset(str_id, 0xFF, sizeof(str_id)); // -1
        strs.clear();
        zhash = c->empty_hash;
    }

    void place_stone(int player, int point) {
        int adj_same[4], n_same = 0;
        int adj_opp [4], n_opp  = 0;
        PointSet libs;
        const auto& ni = cfg->neighbors[point];
        for (int i = 0; i < ni.cnt; i++) {
            int nb = ni.nb[i];
            if (color_at[nb] == EMPTY) { libs.set(nb); }
            else if (color_at[nb] == player) {
                int sid = str_id[nb];
                bool dup = false;
                for (int j = 0; j < n_same; j++) if (adj_same[j] == sid) { dup = true; break; }
                if (!dup) adj_same[n_same++] = sid;
            } else {
                int sid = str_id[nb];
                bool dup = false;
                for (int j = 0; j < n_opp; j++) if (adj_opp[j] == sid) { dup = true; break; }
                if (!dup) adj_opp[n_opp++] = sid;
            }
        }
        int new_sid = (int)strs.size();
        strs.push_back({(int8_t)player, {}, {}});
        GoString& ns = strs[new_sid];
        ns.stones.set(point);
        ns.liberties = libs;

        for (int i = 0; i < n_same; i++) {
            ns.stones    |= strs[adj_same[i]].stones;
            ns.liberties |= strs[adj_same[i]].liberties;
            strs[adj_same[i]].color = 0;
        }
        ns.liberties = ns.liberties - ns.stones;
        ns.stones.for_each([&](int p) { color_at[p] = (int8_t)player; str_id[p] = (int16_t)new_sid; });

        zhash ^= cfg->zobrist[point][player - 1];

        for (int i = 0; i < n_opp; i++) {
            int sid = adj_opp[i];
            if (strs[sid].color == 0) continue;
            strs[sid].liberties.clr(point);
            if (strs[sid].liberties.empty()) remove_string(sid);
        }
    }

    // Compute Zobrist hash that would result from placing player's stone at point,
    // without modifying the board. Accounts for any captured opponent strings.
    // Used by move_violates_ko to avoid copying the entire board.
    uint64_t hash_after_move(int player, int point) const {
        uint64_t h = zhash ^ cfg->zobrist[point][player - 1];
        const auto& ni = cfg->neighbors[point];
        int seen[4]; int ns = 0;
        for (int i = 0; i < ni.cnt; i++) {
            int nb = ni.nb[i];
            if (color_at[nb] == EMPTY || color_at[nb] == player) continue;
            int sid = str_id[nb];
            bool dup = false;
            for (int j = 0; j < ns; j++) if (seen[j] == sid) { dup = true; break; }
            if (dup) continue;
            seen[ns++] = sid;
            if (strs[sid].liberties.count() == 1) {
                strs[sid].stones.for_each([&](int p) {
                    h ^= cfg->zobrist[p][color_at[nb] - 1];
                });
            }
        }
        return h;
    }

    bool is_self_capture(int player, int point) const {
        const auto& ni = cfg->neighbors[point];
        int fsid[4]; int nf = 0;
        for (int i = 0; i < ni.cnt; i++) {
            int nb = ni.nb[i];
            if (color_at[nb] == EMPTY) return false;
            if (color_at[nb] == player) {
                int s = str_id[nb]; bool dup = false;
                for (int j = 0; j < nf; j++) if (fsid[j] == s) { dup = true; break; }
                if (!dup) fsid[nf++] = s;
            } else {
                if (strs[str_id[nb]].liberties.count() == 1) return false;
            }
        }
        for (int i = 0; i < nf; i++)
            if (strs[fsid[i]].liberties.count() != 1) return false;
        return true;
    }

    bool will_capture(int player, int point) const {
        const auto& ni = cfg->neighbors[point];
        for (int i = 0; i < ni.cnt; i++) {
            int nb = ni.nb[i];
            if (color_at[nb] != EMPTY && color_at[nb] != player &&
                strs[str_id[nb]].liberties.count() == 1) return true;
        }
        return false;
    }

private:
    void remove_string(int sid) {
        int c = strs[sid].color;
        strs[sid].stones.for_each([&](int p) {
            const auto& ni = cfg->neighbors[p];
            for (int i = 0; i < ni.cnt; i++) {
                int nb_sid = str_id[ni.nb[i]];
                if (nb_sid >= 0 && nb_sid != sid) strs[nb_sid].liberties.set(p);
            }
            color_at[p] = EMPTY;
            str_id[p]   = -1;
            zhash ^= cfg->zobrist[p][c - 1];
        });
        strs[sid].color = 0;
    }
};

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------
enum MoveType { MOVE_PLAY, MOVE_PASS, MOVE_RESIGN, MOVE_NONE };

struct Move {
    MoveType type;
    int point;
    static Move play(int p)  { return {MOVE_PLAY, p}; }
    static Move pass()       { return {MOVE_PASS, -1}; }
    static Move resign()     { return {MOVE_RESIGN, -1}; }
    static Move none()       { return {MOVE_NONE, -1}; }
    bool is_play()   const { return type == MOVE_PLAY; }
    bool is_pass()   const { return type == MOVE_PASS; }
    bool is_resign() const { return type == MOVE_RESIGN; }
};

// ---------------------------------------------------------------------------
// Situation encoding for superko
// ---------------------------------------------------------------------------
static uint64_t encode_sit(int player, uint64_t hash) {
    return hash ^ (player == BLACK ? 0xAAAAAAAAAAAAAAAAULL : 0x5555555555555555ULL);
}

static bool sit_contains(const PrevStates& v, uint64_t s) {
    for (int i = 0; i < v.cnt; i++) if (v.data[i] == s) return true;
    return false;
}

// No Board copy: hash is computed incrementally via hash_after_move.
static bool move_violates_ko(const Board& board, int player, int point,
                             const PrevStates& prev) {
    if (board.color_at[point] != EMPTY) return false;
    if (!board.will_capture(player, point)) return false;
    return sit_contains(prev, encode_sit(other_color(player),
                                         board.hash_after_move(player, point)));
}

// ---------------------------------------------------------------------------
// Area scoring (Chinese rules): stones + territory flood-fill + komi.
// Used for rollout evaluation and final scoring. Does not handle dead stones;
// assumes the game has been played to a fully resolved position.
// ---------------------------------------------------------------------------
static int area_winner(const Board& bd, double komi = 7.5) {
    int N = bd.cfg->N;
    int black = 0, white = 0;
    std::vector<bool> visited(N, false);

    for (int i = 0; i < N; i++) {
        if      (bd.color_at[i] == BLACK) black++;
        else if (bd.color_at[i] == WHITE) white++;
    }

    for (int i = 0; i < N; i++) {
        if (visited[i] || bd.color_at[i] != EMPTY) continue;
        std::vector<int> region;
        bool touches_black = false, touches_white = false;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            int p = stack.back(); stack.pop_back();
            if (visited[p]) continue;
            if (bd.color_at[p] == BLACK) { touches_black = true; continue; }
            if (bd.color_at[p] == WHITE) { touches_white = true; continue; }
            visited[p] = true;
            region.push_back(p);
            const auto& ni = bd.cfg->neighbors[p];
            for (int j = 0; j < ni.cnt; j++) stack.push_back(ni.nb[j]);
        }
        if      (touches_black && !touches_white) black += (int)region.size();
        else if (touches_white && !touches_black) white += (int)region.size();
    }
    return (black > white + komi) ? BLACK : WHITE;
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------
struct GameState {
    Board board;
    int   next_player = BLACK;
    Move  last_move       = Move::none();
    Move  prev_last_move  = Move::none();
    PrevStates prev_states;

    static GameState new_game(const BoardConfig* cfg) {
        GameState g;
        g.board.init(cfg);
        g.next_player = BLACK;
        return g;
    }

    GameState apply_move(Move m) const {
        GameState g;
        g.board = board;
        if (m.is_play()) g.board.place_stone(next_player, m.point);
        g.next_player    = other_color(next_player);
        g.last_move      = m;
        g.prev_last_move = last_move;
        g.prev_states    = prev_states;
        g.prev_states.push_back(encode_sit(next_player, board.zhash));
        return g;
    }

    bool is_over() const {
        if (last_move.type == MOVE_NONE) return false;
        if (last_move.is_resign()) return true;
        if (prev_last_move.type == MOVE_NONE) return false;
        return last_move.is_pass() && prev_last_move.is_pass();
    }

    bool is_valid_move(Move m) const {
        if (is_over()) return false;
        if (m.is_pass() || m.is_resign()) return true;
        int p = m.point;
        return board.color_at[p] == EMPTY &&
               !board.is_self_capture(next_player, p) &&
               !move_violates_ko(board, next_player, p, prev_states);
    }

    std::vector<Move> legal_moves() const {
        std::vector<Move> mv;
        mv.reserve(board.cfg->N + 2);
        for (int i = 0; i < board.cfg->N; i++)
            if (board.color_at[i] == EMPTY &&
                !board.is_self_capture(next_player, i) &&
                !move_violates_ko(board, next_player, i, prev_states))
                mv.push_back(Move::play(i));
        mv.push_back(Move::pass());
        mv.push_back(Move::resign());
        return mv;
    }

    int winner() const {
        if (last_move.is_resign()) return next_player;
        return area_winner(board);
    }
};

// ---------------------------------------------------------------------------
// Eye detection  (mirrors alphigo/agent/helpers.py)
// ---------------------------------------------------------------------------
static bool is_eye(const Board& bd, int pt, int color) {
    if (bd.color_at[pt] != EMPTY) return false;
    const auto& ni = bd.cfg->neighbors[pt];
    for (int i = 0; i < ni.cnt; i++)
        if (bd.color_at[ni.nb[i]] != color) return false;

    int row = pt / bd.cfg->cols, col = pt % bd.cfg->cols;
    int friendly = 0, offboard = 0;
    int dr[] = {-1,-1, 1, 1}, dc[] = {-1, 1,-1, 1};
    for (int i = 0; i < 4; i++) {
        int cr = row + dr[i], cc = col + dc[i];
        if (cr < 0 || cr >= bd.cfg->rows || cc < 0 || cc >= bd.cfg->cols) offboard++;
        else if (bd.color_at[cr * bd.cfg->cols + cc] == color) friendly++;
    }
    return offboard > 0 ? (offboard + friendly == 4) : (friendly >= 3);
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------
struct MCSTNode {
    GameState state;
    MCSTNode* parent;
    Move move;
    int wins[3]{}; // indexed by color
    int rollouts = 0;
    std::vector<std::unique_ptr<MCSTNode>> children;
    std::vector<Move> unvisited;

    MCSTNode(GameState s, MCSTNode* p, Move m)
        : state(std::move(s)), parent(p), move(m) {
        unvisited = state.legal_moves();
    }

    bool can_expand() const { return !unvisited.empty(); }
    bool is_terminal() const { return state.is_over(); }
    double win_frac(int c) const { return rollouts ? (double)wins[c] / rollouts : 0.0; }

    MCSTNode* expand(std::mt19937& rng) {
        std::uniform_int_distribution<int> d(0, (int)unvisited.size() - 1);
        int idx = d(rng);
        Move m = unvisited[idx];
        unvisited[idx] = unvisited.back(); unvisited.pop_back();
        auto child = std::make_unique<MCSTNode>(state.apply_move(m), this, m);
        MCSTNode* ptr = child.get();
        children.push_back(std::move(child));
        return ptr;
    }
};

static int simulate_random(const GameState& init, std::mt19937& rng,
                           std::vector<int>& buf) {
    Board board = init.board;
    int player  = init.next_player;
    Move last   = init.last_move;
    Move prev   = init.prev_last_move;
    PrevStates psit = init.prev_states;

    auto game_over = [&]() {
        if (last.type == MOVE_NONE) return false;
        if (last.is_resign()) return true;
        if (prev.type == MOVE_NONE) return false;
        return last.is_pass() && prev.is_pass();
    };

    while (!game_over()) {
        buf.clear();
        int N = board.cfg->N;
        for (int i = 0; i < N; i++) {
            if (board.color_at[i] != EMPTY) continue;
            if (board.is_self_capture(player, i)) continue;
            if (move_violates_ko(board, player, i, psit)) continue;
            if (is_eye(board, i, player)) continue;
            buf.push_back(i);
        }
        Move m;
        if (buf.empty()) m = Move::pass();
        else {
            std::uniform_int_distribution<int> d(0, (int)buf.size() - 1);
            m = Move::play(buf[d(rng)]);
        }
        psit.push_back(encode_sit(player, board.zhash));
        if (m.is_play()) board.place_stone(player, m.point);
        prev = last; last = m; player = other_color(player);
    }
    if (last.is_resign()) return player; // other player wins
    return area_winner(board);
}

static Move mcts_select(const GameState& root_state, int rounds, double temp,
                        std::mt19937& rng, float* policy_out = nullptr) {
    MCSTNode root(root_state, nullptr, Move::none());
    int N = root_state.board.cfg->N;
    std::vector<int> rollout_buf;
    rollout_buf.reserve(N);

    for (int i = 0; i < rounds; i++) {
        MCSTNode* node = &root;
        while (!node->can_expand() && !node->is_terminal()) {
            int total = node->rollouts;
            if (total == 0) {
                std::uniform_int_distribution<int> d(0, (int)node->children.size()-1);
                node = node->children[d(rng)].get();
            } else {
                double log_t = std::log((double)total);
                double best_s = -1; MCSTNode* best_c = nullptr;
                int col = node->state.next_player;
                for (auto& c : node->children) {
                    double w = c->win_frac(col);
                    double n = std::max(c->rollouts, 1);
                    double s = w + temp * std::sqrt(log_t / n);
                    if (s > best_s) { best_s = s; best_c = c.get(); }
                }
                node = best_c;
            }
        }
        if (node->can_expand()) node = node->expand(rng);
        int winner = simulate_random(node->state, rng, rollout_buf);
        for (MCSTNode* n = node; n; n = n->parent) { n->wins[winner]++; n->rollouts++; }
    }

    if (policy_out) {
        memset(policy_out, 0, sizeof(float) * N);
        int total_play_visits = 0;
        for (auto& c : root.children)
            if (c->move.is_play()) total_play_visits += c->rollouts;
        if (total_play_visits > 0)
            for (auto& c : root.children)
                if (c->move.is_play())
                    policy_out[c->move.point] = (float)c->rollouts / (float)total_play_visits;
    }

    Move best = Move::none();
    double best_f = -1;
    int col = root_state.next_player;
    for (auto& c : root.children) {
        double f = c->win_frac(col);
        if (f > best_f) { best_f = f; best = c->move; }
    }
    if (best_f < 0.05) best = Move::resign();
    if (best.type == MOVE_NONE) {
        auto lm = root_state.legal_moves();
        if (!lm.empty()) { std::uniform_int_distribution<int> d(0,(int)lm.size()-1); best=lm[d(rng)]; }
        else best = Move::pass();
    }
    return best;
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------
enum Encoder { ENC_ONEPLANE, ENC_SEVENPLANE, ENC_ELEVENPLANE, ENC_ALPHAZERO };

static int enc_planes(Encoder e) {
    switch (e) {
        case ENC_ONEPLANE: return 1; case ENC_SEVENPLANE: return 7;
        case ENC_ELEVENPLANE: return 11; case ENC_ALPHAZERO: return 17;
        default: return 1;
    }
}

static const char* enc_name(Encoder e) {
    switch (e) {
        case ENC_ONEPLANE: return "oneplane"; case ENC_SEVENPLANE: return "sevenplane";
        case ENC_ELEVENPLANE: return "elevenplane"; case ENC_ALPHAZERO: return "alphazero";
        default: return "oneplane";
    }
}

static void encode_pos(Encoder enc, const Board& bd, int player,
                       const PrevStates& psit, float* out,
                       const BoardHistory* hist = nullptr) {
    int N = bd.cfg->N;
    switch (enc) {
    case ENC_ONEPLANE:
        memset(out, 0, sizeof(float) * N);
        for (int i = 0; i < N; i++) {
            if (bd.color_at[i] == player)               out[i] =  1.0f;
            else if (bd.color_at[i] != EMPTY)           out[i] = -1.0f;
        }
        break;
    case ENC_SEVENPLANE: {
        memset(out, 0, sizeof(float) * 7 * N);
        for (int i = 0; i < N; i++) {
            int c = bd.color_at[i];
            if (c == EMPTY) {
                if (move_violates_ko(bd, player, i, psit)) out[6*N + i] = 1.0f;
            } else {
                int libs = bd.strs[bd.str_id[i]].liberties.count();
                int plane = std::min(3, libs) - 1 + (c == player ? 0 : 3);
                out[plane*N + i] = 1.0f;
            }
        }
        break;
    }
    case ENC_ELEVENPLANE: {
        memset(out, 0, sizeof(float) * 11 * N);
        int turn_plane = (player == BLACK) ? 8 : 9;
        for (int i = 0; i < N; i++) out[turn_plane*N + i] = 1.0f;
        for (int i = 0; i < N; i++) {
            int c = bd.color_at[i];
            if (c == EMPTY) {
                if (move_violates_ko(bd, player, i, psit)) out[10*N + i] = 1.0f;
            } else {
                int libs = bd.strs[bd.str_id[i]].liberties.count();
                int plane = std::min(4, libs) - 1 + (c == WHITE ? 4 : 0);
                out[plane*N + i] = 1.0f;
            }
        }
        break;
    }
    case ENC_ALPHAZERO: {
        memset(out, 0, sizeof(float) * 17 * N);
        if (hist) {
            int opp = other_color(player);
            for (int age = 0; age < BoardHistory::DEPTH; age++) {
                const int8_t* board = hist->get(age);
                if (!board) break;
                for (int i = 0; i < N; i++) {
                    if (board[i] == player) out[age * N + i] = 1.0f;
                    if (board[i] == opp)    out[(8 + age) * N + i] = 1.0f;
                }
            }
        }
        float color_val = (player == BLACK) ? 1.0f : 0.0f;
        for (int i = 0; i < N; i++) out[16 * N + i] = color_val;
        break;
    }
    }
}

// ---------------------------------------------------------------------------
// NumPy .npy writer  (v1 format, float32 little-endian)
// ---------------------------------------------------------------------------
static void write_npy(const char* path, const float* data,
                      const std::vector<int>& shape) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Error: cannot write %s\n", path); return; }

    std::string ss = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) ss += ", ";
        ss += std::to_string(shape[i]);
    }
    if (shape.size() == 1) ss += ",";
    ss += ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + ss + ", }";

    int min_len = (int)hdr.size() + 1;
    int total   = 10 + min_len;
    int pad     = (64 - (total % 64)) % 64;
    uint16_t hl = (uint16_t)(min_len + pad);

    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.c_str(), 1, hdr.size(), f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);

    size_t n = 1; for (int s : shape) n *= (size_t)s;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Directory creation
// ---------------------------------------------------------------------------
static void makedirs(const std::string& p) {
    std::string cur;
    for (char c : p) {
        if (c == '/' || c == '\\') {
            if (!cur.empty()) {
#ifdef _WIN32
                _mkdir(cur.c_str());
#else
                mkdir(cur.c_str(), 0755);
#endif
            }
        }
        cur += c;
    }
    if (!cur.empty()) {
#ifdef _WIN32
        _mkdir(cur.c_str());
#else
        mkdir(cur.c_str(), 0755);
#endif
    }
}

// ---------------------------------------------------------------------------
// Checkpoint helpers
// ---------------------------------------------------------------------------

static std::string enc_list_str(const std::vector<Encoder>& encs) {
    std::string s;
    for (int i = 0; i < (int)encs.size(); i++) {
        if (i) s += ",";
        s += enc_name(encs[i]);
    }
    return s;
}

// Checkpoint format (all little-endian):
//   meta file  : 3 x int64  { games_done, positions_done, num_encoders }
//   label file : positions_done * N  float32
//   feature files (one per encoder): positions_done * planes * N  float32

static void write_checkpoint(const std::string& prefix,
                             const std::vector<Encoder>& encs,
                             const std::vector<std::vector<float>>& X_accum,
                             const std::vector<float>& Y_accum,
                             int64_t games_done, int64_t positions_done) {
    {
        FILE* f = fopen((prefix + "_ckpt_meta.bin").c_str(), "wb");
        if (!f) { fprintf(stderr, "[checkpoint] Cannot write meta file\n"); return; }
        int64_t meta[3] = { games_done, positions_done, (int64_t)encs.size() };
        fwrite(meta, sizeof(int64_t), 3, f);
        fclose(f);
    }
    {
        FILE* f = fopen((prefix + "_ckpt_Y.bin").c_str(), "wb");
        if (f) { fwrite(Y_accum.data(), sizeof(float), Y_accum.size(), f); fclose(f); }
    }
    for (int e = 0; e < (int)encs.size(); e++) {
        std::string path = prefix + "_ckpt_" + enc_name(encs[e]) + "_X.bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (f) { fwrite(X_accum[e].data(), sizeof(float), X_accum[e].size(), f); fclose(f); }
    }
}

static bool load_checkpoint(const std::string& prefix,
                            const std::vector<Encoder>& encs,
                            std::vector<std::vector<float>>& X_accum,
                            std::vector<float>& Y_accum,
                            int64_t& games_done, int64_t& positions_done, int N) {
    FILE* f = fopen((prefix + "_ckpt_meta.bin").c_str(), "rb");
    if (!f) return false;
    int64_t meta[3];
    bool ok = (fread(meta, sizeof(int64_t), 3, f) == 3);
    fclose(f);
    if (!ok || meta[2] != (int64_t)encs.size()) {
        fprintf(stderr, "[checkpoint] Stale or incompatible checkpoint -- ignoring.\n");
        return false;
    }
    games_done = meta[0]; positions_done = meta[1];

    Y_accum.resize((size_t)positions_done * N);
    f = fopen((prefix + "_ckpt_Y.bin").c_str(), "rb");
    if (!f || fread(Y_accum.data(), sizeof(float), Y_accum.size(), f) != Y_accum.size()) {
        if (f) fclose(f);
        fprintf(stderr, "[checkpoint] Cannot read label checkpoint.\n");
        return false;
    }
    fclose(f);

    for (int e = 0; e < (int)encs.size(); e++) {
        int planes = enc_planes(encs[e]);
        X_accum[e].resize((size_t)positions_done * planes * N);
        std::string path = prefix + "_ckpt_" + enc_name(encs[e]) + "_X.bin";
        f = fopen(path.c_str(), "rb");
        if (!f || fread(X_accum[e].data(), sizeof(float), X_accum[e].size(), f) != X_accum[e].size()) {
            if (f) fclose(f);
            fprintf(stderr, "[checkpoint] Cannot read feature checkpoint for %s.\n", enc_name(encs[e]));
            return false;
        }
        fclose(f);
    }
    fprintf(stderr, "[checkpoint] Resumed: %lld games, %lld positions.\n",
            (long long)games_done, (long long)positions_done);
    return true;
}

static void clear_checkpoint(const std::string& prefix, const std::vector<Encoder>& encs) {
    remove((prefix + "_ckpt_meta.bin").c_str());
    remove((prefix + "_ckpt_Y.bin").c_str());
    for (auto& enc : encs)
        remove((prefix + "_ckpt_" + enc_name(enc) + "_X.bin").c_str());
}

// ---------------------------------------------------------------------------
// Ownership computation from final board state
// ---------------------------------------------------------------------------
static void compute_ownership(const Board& bd, int8_t* raw_owner) {
    int N = bd.cfg->N;
    std::vector<bool> visited(N, false);

    for (int i = 0; i < N; i++) {
        if      (bd.color_at[i] == BLACK) raw_owner[i] = 1;
        else if (bd.color_at[i] == WHITE) raw_owner[i] = -1;
        else raw_owner[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        if (visited[i] || bd.color_at[i] != EMPTY) continue;
        std::vector<int> region;
        bool touches_black = false, touches_white = false;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            int p = stack.back(); stack.pop_back();
            if (visited[p]) continue;
            if (bd.color_at[p] == BLACK) { touches_black = true; continue; }
            if (bd.color_at[p] == WHITE) { touches_white = true; continue; }
            visited[p] = true;
            region.push_back(p);
            const auto& ni = bd.cfg->neighbors[p];
            for (int j = 0; j < ni.cnt; j++) stack.push_back(ni.nb[j]);
        }
        int8_t owner = 0;
        if      (touches_black && !touches_white) owner = 1;
        else if (touches_white && !touches_black) owner = -1;
        for (int p : region) raw_owner[p] = owner;
    }
}

// ---------------------------------------------------------------------------
// Single-game generation
// ---------------------------------------------------------------------------
struct GameResult {
    std::vector<std::vector<float>> features; // one entry per encoder
    std::vector<float> labels;
    std::vector<float> policy_targets;    // visit-count distributions [positions * N]
    std::vector<float> value_targets;     // game outcome per position [positions]
    std::vector<float> ownership_targets; // per-point ownership [positions * N]
    std::vector<int>   position_players;  // which color was to play
    std::vector<int16_t> move_indices;
    int positions = 0;
};

static GameResult generate_game(const BoardConfig* cfg, int rounds, int max_moves,
                                double temp, const std::vector<Encoder>& encs,
                                std::mt19937& rng) {
    GameResult res;
    res.features.resize(encs.size());
    int N = cfg->N;

    bool has_alphazero = false;
    for (auto e : encs) if (e == ENC_ALPHAZERO) { has_alphazero = true; break; }

    BoardHistory hist;
    std::vector<float> policy_buf(N);

    GameState game = GameState::new_game(cfg);
    int nmoves = 0;

    while (!game.is_over()) {
        if (has_alphazero) hist.push(game.board.color_at, N);

        Move m = mcts_select(game, rounds, temp, rng,
                             has_alphazero ? policy_buf.data() : nullptr);

        res.move_indices.push_back(m.is_play() ? (int16_t)m.point : (int16_t)-1);

        if (m.is_play()) {
            for (int e = 0; e < (int)encs.size(); e++) {
                int pos_sz = enc_planes(encs[e]) * N;
                size_t off_f = res.features[e].size();
                res.features[e].resize(off_f + pos_sz, 0.0f);
                encode_pos(encs[e], game.board, game.next_player, game.prev_states,
                           &res.features[e][off_f],
                           has_alphazero ? &hist : nullptr);
            }
            size_t off_l = res.labels.size();
            res.labels.resize(off_l + N, 0.0f);
            res.labels[off_l + m.point] = 1.0f;

            if (has_alphazero) {
                res.policy_targets.insert(res.policy_targets.end(),
                                          policy_buf.begin(), policy_buf.end());
                res.position_players.push_back(game.next_player);
            }
            res.positions++;
        }
        game = game.apply_move(m);
        if (++nmoves > max_moves) break;
    }

    if (has_alphazero && res.positions > 0) {
        int game_winner = game.is_over() ? game.winner() : area_winner(game.board);
        for (int pos = 0; pos < res.positions; pos++) {
            float val = (game_winner == res.position_players[pos]) ? 1.0f : -1.0f;
            res.value_targets.push_back(val);
        }

        int8_t raw_owner[MAX_N];
        compute_ownership(game.board, raw_owner);
        for (int pos = 0; pos < res.positions; pos++) {
            int pp = res.position_players[pos];
            for (int i = 0; i < N; i++) {
                float own;
                if (raw_owner[i] == 0) own = 0.0f;
                else if ((raw_owner[i] == 1 && pp == BLACK) ||
                         (raw_owner[i] == -1 && pp == WHITE)) own = 1.0f;
                else own = -1.0f;
                res.ownership_targets.push_back(own);
            }
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int    board_size       = 9;
    int    rounds           = 1000;
    double temperature      = 0.8;
    int    max_moves        = 60;
    int    num_games        = 10;
    std::string out_dir     = "data/mcts";
    double split            = 0.8;
    std::string enc_str     = "oneplane";
    std::string tag         = "mcts";
    int    workers          = 1;
    int    checkpoint_every = 500;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return (i+1 < argc) ? argv[++i] : ""; };
        if      (a == "-b" || a == "--board_size")   board_size       = atoi(next());
        else if (a == "-r" || a == "--rounds")        rounds           = atoi(next());
        else if (a == "-t" || a == "--temperature")   temperature      = atof(next());
        else if (a == "-m" || a == "--max_moves")     max_moves        = atoi(next());
        else if (a == "-n" || a == "--num_games")     num_games        = atoi(next());
        else if (a == "-o" || a == "--output-dir")    out_dir          = next();
        else if (a == "--split")                      split            = atof(next());
        else if (a == "-e" || a == "--encoder")       enc_str          = next();
        else if (a == "-w" || a == "--workers")       workers          = atoi(next());
        else if (a == "--tag")                        tag              = next();
        else if (a == "--checkpoint-every")           checkpoint_every = atoi(next());
        else if (a == "-h" || a == "--help") {
            printf(
                "Usage: generate_mcst_games [options]\n"
                "  -b  --board_size       INT    board size (default 9)\n"
                "  -r  --rounds           INT    MCTS rounds/move (default 1000)\n"
                "  -t  --temperature      FLOAT  UCT exploration constant (default 0.8)\n"
                "  -m  --max_moves        INT    move cap per game (default 60)\n"
                "  -n  --num_games        INT    total games to generate (default 10)\n"
                "  -o  --output-dir       STR    output directory (default data/mcts)\n"
                "  --split                FLOAT  train/test fraction (default 0.8)\n"
                "  -e  --encoder          STR    comma-separated encoder list:\n"
                "                                oneplane,sevenplane,elevenplane,alphazero\n"
                "                                (alphazero: 17-plane + policy/value/ownership)\n"
                "  -w  --workers          INT    parallel workers (default 1)\n"
                "  --tag                  STR    filename prefix tag (default: mcts)\n"
                "  --checkpoint-every     INT    save checkpoint every N games (default 500)\n"
                "\n"
                "Output files:\n"
                "  {out}/{tag}_{encoder}_features_train.npy  (one per encoder)\n"
                "  {out}/{tag}_labels_train.npy               (shared across encoders)\n"
                "\n"
                "Example — multi-encoder with checkpointing:\n"
                "  generate_mcst_games -b 9 -r 1000 -n 500 -w 4 \\\n"
                "      -e oneplane,sevenplane,elevenplane --checkpoint-every 50\n");
            return 0;
        }
    }

    // ---------- parse encoder list (comma-separated) ----------
    std::vector<Encoder> encs;
    {
        std::istringstream iss(enc_str);
        std::string tok;
        while (std::getline(iss, tok, ',')) {
            if (tok == "sevenplane")       encs.push_back(ENC_SEVENPLANE);
            else if (tok == "elevenplane") encs.push_back(ENC_ELEVENPLANE);
            else if (tok == "alphazero")   encs.push_back(ENC_ALPHAZERO);
            else                           encs.push_back(ENC_ONEPLANE);
        }
        if (encs.empty()) encs.push_back(ENC_ONEPLANE);
    }

    BoardConfig cfg;
    cfg.init(board_size, board_size);
    int N = cfg.N;

    makedirs(out_dir);
    std::string prefix = out_dir + "/" + tag;

    // ---------- per-batch streaming: X_accum holds ONE batch at a time ----------
    // After each batch the vectors are written to a numbered .npy file and
    // cleared, so peak RAM is ~1 batch worth of features + MCTS trees.
    // A Python script (merge_batches.py) concatenates, shuffles and splits.
    bool has_alphazero = false;
    for (auto e : encs) if (e == ENC_ALPHAZERO) { has_alphazero = true; break; }

    std::vector<std::vector<float>> X_accum(encs.size());
    std::vector<float> Y_accum;
    std::vector<float> P_accum, V_accum, O_accum;
    std::vector<std::vector<int16_t>> game_records;
    int64_t games_done = 0, positions_done = 0;
    int batch_num = 0;

    // Resume: only read the meta (games_done / positions_done).
    // Per-batch .npy files from previous runs are already on disk.
    {
        FILE* f = fopen((prefix + "_ckpt_meta.bin").c_str(), "rb");
        if (f) {
            int64_t meta[3]{};
            if (fread(meta, sizeof(int64_t), 3, f) == 3) {
                games_done     = meta[0];
                positions_done = meta[1];
                batch_num      = (int)(games_done / checkpoint_every);
                fprintf(stderr, "[checkpoint] Resumed: %lld games done, batch_num=%d\n",
                        (long long)games_done, batch_num);
            }
            fclose(f);
        }
    }

    int remaining = num_games - (int)games_done;
    if (remaining > 0)
        fprintf(stderr,
                "Generating %d/%d games  encoders=[%s]  %dx%d  %d rounds  "
                "%d workers  ckpt every %d\n",
                remaining, num_games, enc_list_str(encs).c_str(),
                board_size, board_size, rounds, workers, checkpoint_every);
    else
        fprintf(stderr, "[checkpoint] All %d games already done.\n", num_games);

    // ---------- generate games in batches, writing .npy per batch ----------
    while ((int)games_done < num_games) {
        int batch = std::min(checkpoint_every, num_games - (int)games_done);
        int64_t pos_before = positions_done;

        // Reserve for this batch only.
        {
            double avg_pos = (games_done > 0)
                ? (double)positions_done / games_done
                : (double)max_moves * 0.5;
            int64_t est = (int64_t)(avg_pos * batch) + 1000;
            for (int e = 0; e < (int)encs.size(); e++) {
                size_t cap = (size_t)est * enc_planes(encs[e]) * N;
                if (X_accum[e].capacity() < cap) X_accum[e].reserve(cap);
            }
            if (Y_accum.capacity() < (size_t)est * N) Y_accum.reserve((size_t)est * N);
        }
        std::vector<GameResult> results(batch);
        std::atomic<int> completed_batch{0};

        fprintf(stderr, "  Batch: games %lld-%lld of %d ...\n",
                (long long)games_done + 1, (long long)games_done + batch, num_games);

#ifdef _WIN32
        struct WorkerCtx {
            std::vector<GameResult>*    results;
            const BoardConfig*          cfg;
            int rounds, max_moves, batch, workers, tid;
            double temperature;
            const std::vector<Encoder>* encs;
            std::atomic<int>*           completed;
            CRITICAL_SECTION*           cs;
        };

        CRITICAL_SECTION cs;
        InitializeCriticalSection(&cs);

        struct WorkerRunner {
            static unsigned __stdcall run(void* arg) {
                auto* ctx = (WorkerCtx*)arg;
                uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                ^ ((uint64_t)ctx->tid * 6364136223846793005ULL);
std::mt19937 rng((uint32_t)(seed ^ (seed >> 32)));
                for (int g = ctx->tid; g < ctx->batch; g += ctx->workers) {
                    (*ctx->results)[g] = generate_game(ctx->cfg, ctx->rounds, ctx->max_moves,
                                                       ctx->temperature, *ctx->encs, rng);
                    int c = ++(*ctx->completed);
                    EnterCriticalSection(ctx->cs);
                    fprintf(stderr, "\r    Game %d/%d  (%d positions)  ",
                            c, ctx->batch, (*ctx->results)[g].positions);
                    LeaveCriticalSection(ctx->cs);
                }
                return 0;
            }
        };

        std::vector<WorkerCtx> ctxs(workers);
        std::vector<HANDLE> handles;
        for (int t = 0; t < workers; t++)
            ctxs[t] = {&results, &cfg, rounds, max_moves, batch, workers, t,
                       temperature, &encs, &completed_batch, &cs};

        if (workers <= 1) {
            WorkerRunner::run(&ctxs[0]);
        } else {
            for (int t = 0; t < workers; t++) {
                HANDLE h = (HANDLE)_beginthreadex(nullptr, 0, WorkerRunner::run, &ctxs[t], 0, nullptr);
                handles.push_back(h);
            }
            WaitForMultipleObjects((DWORD)handles.size(), handles.data(), TRUE, INFINITE);
            for (auto h : handles) CloseHandle(h);
        }
        DeleteCriticalSection(&cs);
#else
        std::mutex io_mtx;
        auto worker_fn = [&](int tid) {
            uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                ^ ((uint64_t)tid * 6364136223846793005ULL);
std::mt19937 rng((uint32_t)(seed ^ (seed >> 32)));
            for (int g = tid; g < batch; g += workers) {
                results[g] = generate_game(&cfg, rounds, max_moves, temperature, encs, rng);
                int c = ++completed_batch;
                { std::lock_guard<std::mutex> lk(io_mtx);
                  fprintf(stderr, "\r    Game %d/%d  (%d positions)  ",
                          c, batch, results[g].positions); }
            }
        };
        if (workers <= 1) {
            worker_fn(0);
        } else {
            std::vector<std::thread> threads;
            for (int t = 0; t < workers; t++) threads.emplace_back(worker_fn, t);
            for (auto& t : threads) t.join();
        }
#endif
        fprintf(stderr, "\n");

        // Accumulate batch results and free per-game memory immediately.
        for (auto& r : results) {
            for (int e = 0; e < (int)encs.size(); e++)
                X_accum[e].insert(X_accum[e].end(),
                                  r.features[e].begin(), r.features[e].end());
            Y_accum.insert(Y_accum.end(), r.labels.begin(), r.labels.end());
            if (has_alphazero) {
                P_accum.insert(P_accum.end(), r.policy_targets.begin(), r.policy_targets.end());
                V_accum.insert(V_accum.end(), r.value_targets.begin(), r.value_targets.end());
                O_accum.insert(O_accum.end(), r.ownership_targets.begin(), r.ownership_targets.end());
                r.policy_targets.clear(); r.policy_targets.shrink_to_fit();
                r.value_targets.clear(); r.value_targets.shrink_to_fit();
                r.ownership_targets.clear(); r.ownership_targets.shrink_to_fit();
            }
            positions_done += r.positions;
            game_records.push_back(std::move(r.move_indices));
            for (auto& fv : r.features) { fv.clear(); fv.shrink_to_fit(); }
            r.labels.clear(); r.labels.shrink_to_fit();
        }
        games_done += (int64_t)batch;

        // Write this batch as numbered .npy files and free memory immediately.
        {
            char bn[32]; snprintf(bn, sizeof(bn), "_b%04d", ++batch_num);
            int64_t bpos = positions_done - pos_before;
            for (int e = 0; e < (int)encs.size(); e++) {
                int planes = enc_planes(encs[e]);
                std::string fp = out_dir + "/" + tag + bn + "_" + enc_name(encs[e]) + "_features.npy";
                write_npy(fp.c_str(), X_accum[e].data(),
                          {(int)bpos, planes, board_size, board_size});
                X_accum[e].clear(); X_accum[e].shrink_to_fit();
            }
            std::string lp = out_dir + "/" + tag + bn + "_labels.npy";
            write_npy(lp.c_str(), Y_accum.data(), {(int)bpos, N});
            Y_accum.clear(); Y_accum.shrink_to_fit();

            if (has_alphazero) {
                std::string pp = out_dir + "/" + tag + bn + "_policy.npy";
                write_npy(pp.c_str(), P_accum.data(), {(int)bpos, N});
                P_accum.clear(); P_accum.shrink_to_fit();

                std::string vp = out_dir + "/" + tag + bn + "_values.npy";
                write_npy(vp.c_str(), V_accum.data(), {(int)bpos});
                V_accum.clear(); V_accum.shrink_to_fit();

                std::string op = out_dir + "/" + tag + bn + "_ownership.npy";
                write_npy(op.c_str(), O_accum.data(), {(int)bpos, N});
                O_accum.clear(); O_accum.shrink_to_fit();
            }
        }

        // Write meta checkpoint so we can resume from the last completed batch.
        {
            FILE* f = fopen((prefix + "_ckpt_meta.bin").c_str(), "wb");
            if (f) {
                int64_t meta[3] = { games_done, positions_done, (int64_t)encs.size() };
                fwrite(meta, sizeof(int64_t), 3, f);
                fclose(f);
            }
        }
        fprintf(stderr, "[checkpoint] %lld/%d games done  %lld positions  batch=%d\n",
                (long long)games_done, num_games, (long long)positions_done, batch_num);
    }

    // Remove meta checkpoint — generation is complete.
    remove((prefix + "_ckpt_meta.bin").c_str());

    // Write game records binary file for Python re-encoding.
    // Format (all little-endian): int32 total_games,
    //   per game: int16 num_moves, int16[num_moves] move indices
    //     play move = row * board_size + col,  pass = -1
    {
        std::string rec_path = out_dir + "/" + tag + "_game_records.bin";
        FILE* f = fopen(rec_path.c_str(), "wb");
        if (f) {
            int32_t total = (int32_t)game_records.size();
            fwrite(&total, sizeof(int32_t), 1, f);
            for (auto& moves : game_records) {
                int16_t n = (int16_t)moves.size();
                fwrite(&n, sizeof(int16_t), 1, f);
                if (n > 0)
                    fwrite(moves.data(), sizeof(int16_t), (size_t)n, f);
            }
            fclose(f);
            fprintf(stderr, "[records] Saved %d game records to %s\n",
                    total, rec_path.c_str());
        } else {
            fprintf(stderr, "[records] WARNING: Could not write %s\n",
                    rec_path.c_str());
        }
    }

    printf("\nDone. %lld positions written across %d batch files in %s/\n",
           (long long)positions_done, batch_num, out_dir.c_str());
    printf("Run merge_batches.py to concatenate, shuffle and split:\n");
    printf("  python merge_batches.py %s --tag %s\n", out_dir.c_str(), tag.c_str());
    return 0;
}
