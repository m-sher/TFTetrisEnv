#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================
// Constants
// ============================================================

#define BOARD_ROWS 40
#define VISIBLE_ROWS 20
#define BOARD_COLS 10
#define ROTATIONS 4
#define SPIN_STATES 2

// Keys (same as pathfinder.c / Moves.py)
#define KEY_START 0
#define KEY_HOLD 1
#define KEY_TAP_LEFT 2
#define KEY_TAP_RIGHT 3
#define KEY_DAS_LEFT 4
#define KEY_DAS_RIGHT 5
#define KEY_CLOCKWISE 6
#define KEY_ANTICLOCKWISE 7
#define KEY_ROTATE_180 8
#define KEY_SOFT_DROP 9
#define KEY_HARD_DROP 10
#define KEY_PAD 11

// Piece Types
#define PIECE_N 0
#define PIECE_I 1
#define PIECE_J 2
#define PIECE_L 3
#define PIECE_O 4
#define PIECE_S 5
#define PIECE_T 6
#define PIECE_Z 7

// Spin types (matching Scorer.py Spins enum)
#define SPIN_NONE 0
#define SPIN_T_MINI 1
#define SPIN_T_FULL 2
#define SPIN_ALL_MINI 3

// Garbage row marker: bit 10 set on a uint16_t row means the row is
// simulated garbage during search — it is treated as occupied but
// cannot be cleared by clear_lines().
#define GARB_ROW_MARKER (1u << 10)  // 0x0400

// BFS limits
#define BFS_QUEUE_CAPACITY 8192
#define BFS_STATE_SPACE (BOARD_ROWS * BOARD_COLS * ROTATIONS)

// Placement limits
#define MAX_PLACEMENTS 512

// Beam search limits
#define MAX_BEAM_WIDTH 256
#define MAX_SEARCH_DEPTH 16

// ============================================================
// Piece / Orientation structures (same as pathfinder.c)
// ============================================================

typedef struct {
    uint16_t row_masks[4];
    int min_col;
    int max_col;
    int min_row;
    int max_row;
    int row_offsets[4];
} PieceOrientation;

typedef struct {
    PieceOrientation orientations[4];
} PieceDef;

// BFS state tracking (for internal placement finder)
typedef struct {
    int16_t parent;
    int8_t last_move;
    int16_t depth;
    int8_t delta_r;
    int8_t delta_row;
    int8_t delta_col;
} BFSStateMeta;

// A single placement result from the internal BFS
typedef struct {
    int rot;
    int col;
    int landing_row;
    int spin_type;      // SPIN_NONE, SPIN_T_MINI, SPIN_T_FULL, SPIN_ALL_MINI
    int delta_r;        // Rotation delta (for scorer logic)
    int delta_loc_sum;  // abs(delta_row) + abs(delta_col) for T-spin mini/full distinction
    int bfs_state;      // BFS state index (for key sequence reconstruction)
} Placement;

// Search state for beam search
typedef struct {
    uint16_t board[BOARD_ROWS];
    int b2b;
    int combo;
    float total_attack;
    float max_single_attack;
    float b2b_attack;          // Attack only from b2b-maintaining clears (spins/tetrises/PCs)
    float max_b2b_attack;      // Largest single b2b-maintaining attack in this path
    int total_lines_cleared;   // Total lines cleared in this search path
    int hold_piece;
    int next_queue_idx;
    int depth0_placement_idx;  // Which placement was chosen at depth 0 (for output)
    bool b2b_broken;
    int prev_b2b;
    float streak_attack;     // Cumulative attack across current unbroken combo (resets on 0-attack placement)
    int garbage_remaining;   // Simulated garbage rows not yet pushed
    int garbage_timer;       // Ticks until next garbage push (decremented on non-clear)
    uint8_t bag_seen;        // Bitmask of pieces consumed from current 7-bag (bits 1-7)
    float score;
} SearchState;

// ============================================================
// Bag Tracking Helpers (for speculative search beyond known queue)
// ============================================================

// Update bag tracking when a piece is consumed.
// Returns the new bag_seen bitmask.
// When all 7 pieces (PIECE_I=1 through PIECE_Z=7) have been consumed,
// resets to 0 to represent a fresh bag.
static uint8_t bag_consume_piece(uint8_t bag_seen, int piece_type) {
    bag_seen |= (uint8_t)(1 << piece_type);
    // Bits 1-7 all set = 0xFE means all 7 pieces consumed
    if ((bag_seen & 0xFE) == 0xFE) bag_seen = 0;
    return bag_seen;
}

// Get the remaining pieces in the current bag.
// Writes piece types to out_pieces[], returns count.
static int bag_get_remaining(uint8_t bag_seen, int* out_pieces) {
    int count = 0;
    for (int p = PIECE_I; p <= PIECE_Z; p++) {
        if (!(bag_seen & (1 << p))) out_pieces[count++] = p;
    }
    return count;
}

// ============================================================
// Globals (static to this translation unit)
// ============================================================

static PieceDef B2B_PIECES[8];
static bool b2b_initialized = false;

// Last search result — read by the C game loop to avoid lossy action_idx decode.
// b2b_search_c() writes this; b2b_run_eval_games() reads it.
static Placement b2b_last_placement;

// Kick tables: [from_rot][to_rot][kick_index][0=row, 1=col]
static int8_t B2B_KICKS[4][4][5][2];
static int8_t B2B_I_KICKS[4][4][5][2];

// Heuristic weights — SURVIVAL first, then B2B, then attack
// These are mutable so they can be tuned at runtime via b2b_set_weights().
static float W_HEIGHT      = 6.0f;
static float W_AVG_HEIGHT  = 1.5f;
static float W_BUMPINESS   = 1.2f;
static float W_HOLES       = 4.0f;      // reduced: holes aren't always bad (spin setups)
static float W_HOLE_COL    = 2.5f;      // penalty per distinct column with holes
static float W_DEEP_HOLES  = 3.0f;      // extra penalty for holes buried deep
static float W_CLEARABLE   = 2.5f;
static float W_B2B         = 100.0f;
static float W_COMBO       = 4.0f;
static float W_B2B_BREAK   = 18.0f;      // strong: only break in danger zone
static float W_SPIKE       = 4.0f;
static float W_SPIN_SETUP  = 10.0f;      // reward for immobile spin placements
static float W_TSLOT       = 8.0f;      // reward for T-spin slots specifically
static float W_IMMOBILE_CLEAR = 8.0f;   // reward for immobile placements that clear lines (b2b downstack)
static float W_HOLE_CEILING = 0.8f;     // penalty for filled cells above enclosed holes
static float W_WASTED_HOLE = 8.0f;      // penalty for non-enclosed holes not in any immobile placement
static float W_ATTACK      = 5.0f;     // linear reward per point of total attack in search path
static float W_APP_BONUS   = 0.0f;    // reward for high attack-per-piece efficiency (APP²)
static float W_GARB_CANCEL = 4.0f;    // reward per attack point that cancels pending garbage
static float W_STREAK     = 3.0f;    // superlinear reward for consecutive-attack streak

// Runtime weight setter — allows Python to tune coefficients for testing
void b2b_set_weights(
    float height, float avg_height, float bumpiness,
    float holes, float hole_col, float deep_holes,
    float clearable, float b2b, float combo,
    float b2b_break, float spike, float spin_setup,
    float tslot, float immobile_clear, float hole_ceiling,
    float wasted_hole, float attack, float app_bonus,
    float garb_cancel, float streak
) {
    W_HEIGHT = height;
    W_AVG_HEIGHT = avg_height;
    W_BUMPINESS = bumpiness;
    W_HOLES = holes;
    W_HOLE_COL = hole_col;
    W_DEEP_HOLES = deep_holes;
    W_CLEARABLE = clearable;
    W_B2B = b2b;
    W_COMBO = combo;
    W_B2B_BREAK = b2b_break;
    W_SPIKE = spike;
    W_SPIN_SETUP = spin_setup;
    W_TSLOT = tslot;
    W_IMMOBILE_CLEAR = immobile_clear;
    W_HOLE_CEILING = hole_ceiling;
    W_WASTED_HOLE = wasted_hole;
    W_ATTACK = attack;
    W_APP_BONUS = app_bonus;
    W_GARB_CANCEL = garb_cancel;
    W_STREAK = streak;
}

// ============================================================
// Piece / Kick Initialization (copied from pathfinder.c)
// ============================================================

static void b2b_init_pieces(void) {
    if (b2b_initialized) return;

    memset(B2B_PIECES, 0, sizeof(B2B_PIECES));
    memset(B2B_KICKS, 0, sizeof(B2B_KICKS));
    memset(B2B_I_KICKS, 0, sizeof(B2B_I_KICKS));

    // I Piece
    B2B_PIECES[PIECE_I].orientations[0] = (PieceOrientation){ .row_masks={0, 15, 0, 0}, .min_col=0, .max_col=3, .min_row=1, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[1] = (PieceOrientation){ .row_masks={4, 4, 4, 4}, .min_col=2, .max_col=2, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[2] = (PieceOrientation){ .row_masks={0, 0, 15, 0}, .min_col=0, .max_col=3, .min_row=2, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 2, 2}, .min_col=1, .max_col=1, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };

    // J Piece
    B2B_PIECES[PIECE_J].orientations[0] = (PieceOrientation){ .row_masks={1, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[1] = (PieceOrientation){ .row_masks={6, 2, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 4, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 3, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // L Piece
    B2B_PIECES[PIECE_L].orientations[0] = (PieceOrientation){ .row_masks={4, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[1] = (PieceOrientation){ .row_masks={2, 2, 6, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 1, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[3] = (PieceOrientation){ .row_masks={3, 2, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // O Piece
    PieceOrientation o_orient = (PieceOrientation){ .row_masks={6, 6, 0, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    for (int i = 0; i < 4; i++) B2B_PIECES[PIECE_O].orientations[i] = o_orient;

    // S Piece
    B2B_PIECES[PIECE_S].orientations[0] = (PieceOrientation){ .row_masks={6, 3, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 4, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[2] = (PieceOrientation){ .row_masks={0, 6, 3, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[3] = (PieceOrientation){ .row_masks={1, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // T Piece
    B2B_PIECES[PIECE_T].orientations[0] = (PieceOrientation){ .row_masks={2, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 2, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // Z Piece
    B2B_PIECES[PIECE_Z].orientations[0] = (PieceOrientation){ .row_masks={3, 6, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[1] = (PieceOrientation){ .row_masks={4, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[2] = (PieceOrientation){ .row_masks={0, 3, 6, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 1, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // --- Standard Kicks ---
    int8_t k01[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[0][1][i][0] = k01[i][0]; B2B_KICKS[0][1][i][1] = k01[i][1]; }

    int8_t k03[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[0][3][i][0] = k03[i][0]; B2B_KICKS[0][3][i][1] = k03[i][1]; }

    int8_t k10[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[1][0][i][0] = k10[i][0]; B2B_KICKS[1][0][i][1] = k10[i][1]; }

    int8_t k12[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[1][2][i][0] = k12[i][0]; B2B_KICKS[1][2][i][1] = k12[i][1]; }

    int8_t k21[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[2][1][i][0] = k21[i][0]; B2B_KICKS[2][1][i][1] = k21[i][1]; }

    int8_t k23[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[2][3][i][0] = k23[i][0]; B2B_KICKS[2][3][i][1] = k23[i][1]; }

    int8_t k30[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[3][0][i][0] = k30[i][0]; B2B_KICKS[3][0][i][1] = k30[i][1]; }

    int8_t k32[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[3][2][i][0] = k32[i][0]; B2B_KICKS[3][2][i][1] = k32[i][1]; }

    // 180 Standard Kicks
    int8_t k02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[0][2][i][0] = k02[i][0]; B2B_KICKS[0][2][i][1] = k02[i][1]; }

    int8_t k13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[1][3][i][0] = k13[i][0]; B2B_KICKS[1][3][i][1] = k13[i][1]; }

    int8_t k20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[2][0][i][0] = k20[i][0]; B2B_KICKS[2][0][i][1] = k20[i][1]; }

    int8_t k31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[3][1][i][0] = k31[i][0]; B2B_KICKS[3][1][i][1] = k31[i][1]; }

    // --- I Kicks ---
    int8_t ik01[4][2] = {{0,1}, {0,-2}, {1,-2}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[0][1][i][0] = ik01[i][0]; B2B_I_KICKS[0][1][i][1] = ik01[i][1]; }

    int8_t ik03[4][2] = {{0,-1}, {0,2}, {1,2}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[0][3][i][0] = ik03[i][0]; B2B_I_KICKS[0][3][i][1] = ik03[i][1]; }

    int8_t ik10[4][2] = {{0,-1}, {0,2}, {2,-1}, {-1,2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[1][0][i][0] = ik10[i][0]; B2B_I_KICKS[1][0][i][1] = ik10[i][1]; }

    int8_t ik12[4][2] = {{0,-1}, {0,2}, {-2,-1}, {1,2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[1][2][i][0] = ik12[i][0]; B2B_I_KICKS[1][2][i][1] = ik12[i][1]; }

    int8_t ik21[4][2] = {{0,-2}, {0,1}, {-1,-2}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[2][1][i][0] = ik21[i][0]; B2B_I_KICKS[2][1][i][1] = ik21[i][1]; }

    int8_t ik23[4][2] = {{0,2}, {0,-1}, {-1,2}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[2][3][i][0] = ik23[i][0]; B2B_I_KICKS[2][3][i][1] = ik23[i][1]; }

    int8_t ik30[4][2] = {{0,1}, {0,-2}, {2,1}, {-1,-2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[3][0][i][0] = ik30[i][0]; B2B_I_KICKS[3][0][i][1] = ik30[i][1]; }

    int8_t ik32[4][2] = {{0,1}, {0,-2}, {-2,1}, {1,-2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[3][2][i][0] = ik32[i][0]; B2B_I_KICKS[3][2][i][1] = ik32[i][1]; }

    // 180 I Kicks
    int8_t ik02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[0][2][i][0] = ik02[i][0]; B2B_I_KICKS[0][2][i][1] = ik02[i][1]; }

    int8_t ik13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[1][3][i][0] = ik13[i][0]; B2B_I_KICKS[1][3][i][1] = ik13[i][1]; }

    int8_t ik20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[2][0][i][0] = ik20[i][0]; B2B_I_KICKS[2][0][i][1] = ik20[i][1]; }

    int8_t ik31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[3][1][i][0] = ik31[i][0]; B2B_I_KICKS[3][1][i][1] = ik31[i][1]; }

    b2b_initialized = true;
}

// ============================================================
// Collision / Movement Helpers (adapted from pathfinder.c)
// ============================================================

static int b2b_check_collision(const uint16_t* board_rows, int board_height,
                               int piece_type, int rot, int r, int c) {
    PieceOrientation* ori = &B2B_PIECES[piece_type].orientations[rot];

    if (c + ori->min_col < 0 || c + ori->max_col >= BOARD_COLS) return 1;
    if (r + ori->min_row < 0) return 1;
    if (r + ori->max_row >= board_height) return 1;

    for (int i = 0; i < 4; i++) {
        int board_row = r + i;
        if (board_row < 0 || board_row >= board_height) continue;
        uint16_t mask = ori->row_masks[i];
        uint16_t shifted = (c >= 0) ? (mask << c) : (mask >> (-c));
        if (board_rows[board_row] & shifted) return 1;
    }
    return 0;
}

static int b2b_hard_drop_row(const uint16_t* board_rows, int board_height,
                             int piece_type, int rot, int r, int c) {
    int curr = r;
    while (!b2b_check_collision(board_rows, board_height, piece_type, rot, curr + 1, c)) {
        curr++;
    }
    return curr;
}

static int b2b_encode_state(int r, int c, int rot, int piece_type) {
    int min_col = B2B_PIECES[piece_type].orientations[rot].min_col;
    int norm_col = c + min_col;
    if (norm_col < 0 || norm_col >= BOARD_COLS) return -1;
    if (r < 0 || r >= BOARD_ROWS) return -1;
    return ((r * BOARD_COLS) + norm_col) * 4 + rot;
}

static void b2b_decode_state(int state, int* r, int* c, int* rot, int piece_type) {
    *rot = state % 4;
    int base = state / 4;
    int norm_col = base % BOARD_COLS;
    *r = base / BOARD_COLS;
    int min_col = B2B_PIECES[piece_type].orientations[*rot].min_col;
    *c = norm_col - min_col;
}

// ============================================================
// Spin Detection (adapted from pathfinder.c + Scorer.py)
// ============================================================

// Returns detailed T-spin type: SPIN_NONE, SPIN_T_MINI, or SPIN_T_FULL
static int detect_t_spin(const uint16_t* board, int board_height,
                         int r, int c, int rot, int delta_loc_sum) {
    // 3-corner rule from Scorer.py
    int corners[4][2] = {{0,0}, {0,2}, {2,2}, {2,0}}; // TL, TR, BR, BL
    bool filled[4];

    for (int i = 0; i < 4; i++) {
        int cr = r + corners[i][0];
        int cc = c + corners[i][1];
        if (cr >= board_height || cc < 0 || cc >= BOARD_COLS || cr < 0) {
            filled[i] = true;
        } else {
            filled[i] = (board[cr] & (1 << cc)) != 0;
        }
    }

    int total_filled = 0;
    for (int i = 0; i < 4; i++) if (filled[i]) total_filled++;
    if (total_filled < 3) return SPIN_NONE;

    // Determine front/back corners based on rotation
    // Scorer.py: back cell = the cell NOT in the piece's cells for that rotation
    // Rot 0: back=row2,col1 -> back_idx=3 (corner indices BL=3, TL=0 are back-side)
    // Rot 1: back=row1,col0 -> back_idx=0 (TL=0, BL=3 are back-side)
    // Rot 2: back=row0,col1 -> back_idx=0 (TL=0, TR=1 are back-side)
    // Rot 3: back=row1,col2 -> back_idx=1 (TR=1, BR=2 are back-side)

    // Actually: from Scorer.py, the "back" cell is the one missing from the T shape.
    // T orientations (cells in 3x3):
    // Rot 0: [0,1],[1,0],[1,1],[1,2] -> missing edge cell is [2,1] -> back direction is down
    //   Back corners: BL(2,0)=idx3, BR(2,2)=idx2. Front: TL(0,0)=idx0, TR(0,2)=idx1
    // Rot 1: [0,1],[1,1],[1,2],[2,1] -> missing [1,0] -> back is left
    //   Back corners: TL(0,0)=idx0, BL(2,0)=idx3. Front: TR(0,2)=idx1, BR(2,2)=idx2
    // Rot 2: [1,0],[1,1],[1,2],[2,1] -> missing [0,1] -> back is up
    //   Back corners: TL(0,0)=idx0, TR(0,2)=idx1. Front: BL(2,0)=idx3, BR(2,2)=idx2
    // Rot 3: [0,1],[1,0],[1,1],[2,1] -> missing [1,2] -> back is right
    //   Back corners: TR(0,2)=idx1, BR(2,2)=idx2. Front: TL(0,0)=idx0, BL(2,0)=idx3

    int front_filled = 0, back_filled = 0;
    switch (rot) {
        case 0: // front: TL(0), TR(1); back: BR(2), BL(3)
            front_filled = filled[0] + filled[1];
            back_filled = filled[2] + filled[3];
            break;
        case 1: // front: TR(1), BR(2); back: TL(0), BL(3)
            front_filled = filled[1] + filled[2];
            back_filled = filled[0] + filled[3];
            break;
        case 2: // front: BR(2), BL(3); back: TL(0), TR(1)
            front_filled = filled[2] + filled[3];
            back_filled = filled[0] + filled[1];
            break;
        case 3: // front: TL(0), BL(3); back: TR(1), BR(2)
            front_filled = filled[0] + filled[3];
            back_filled = filled[1] + filled[2];
            break;
    }

    if (front_filled == 2 && back_filled >= 1) {
        return SPIN_T_FULL;
    } else if (front_filled == 1 && back_filled == 2) {
        if (delta_loc_sum > 2) {
            return SPIN_T_FULL; // Kicked far enough
        } else {
            return SPIN_T_MINI;
        }
    }
    return SPIN_NONE;
}

// Check immobility for non-T pieces (ALL_MINI detection)
static bool b2b_check_immobility(const uint16_t* board, int board_height,
                                 int piece_type, int rot, int r, int c) {
    int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    for (int i = 0; i < 4; i++) {
        if (!b2b_check_collision(board, board_height, piece_type, rot,
                                 r + dirs[i][0], c + dirs[i][1])) {
            return false;
        }
    }
    return true;
}

// ============================================================
// Board Simulation
// ============================================================

// Lock a piece onto the board bitmask array
static void lock_piece_on_board(uint16_t* board, int board_height,
                                int piece_type, int rot, int r, int c) {
    PieceOrientation* ori = &B2B_PIECES[piece_type].orientations[rot];
    for (int i = 0; i < 4; i++) {
        int board_row = r + i;
        if (board_row < 0 || board_row >= board_height) continue;
        uint16_t mask = ori->row_masks[i];
        if (mask == 0) continue;
        uint16_t shifted = (c >= 0) ? (mask << c) : (mask >> (-c));
        board[board_row] |= shifted;
    }
}

// Clear full lines, shift rows down. Returns number of lines cleared.
// Rows with GARB_ROW_MARKER set are simulated garbage and are never cleared
// even if all 10 playfield bits are filled.
static int clear_lines(uint16_t* board, int board_height) {
    uint16_t full_row = (1 << BOARD_COLS) - 1; // 0x3FF for 10 cols
    int clears = 0;

    // Iterate bottom-up, shift down when clearing
    int write = board_height - 1;
    for (int read = board_height - 1; read >= 0; read--) {
        if ((board[read] & full_row) == full_row &&
            !(board[read] & GARB_ROW_MARKER)) {
            clears++;
        } else {
            board[write] = board[read];
            write--;
        }
    }
    // Fill remaining top rows with empty
    for (int i = write; i >= 0; i--) {
        board[i] = 0;
    }
    return clears;
}

// Push simulated garbage rows onto the board during search.
// Rows are fully filled (all 10 bits) + GARB_ROW_MARKER so they act as
// unclearable occupied space. Everything above shifts up.
// Returns the number of rows actually pushed (may be less if board would
// overflow and cause instant death).
static int push_simulated_garbage(uint16_t* board, int board_height, int rows) {
    if (rows <= 0) return 0;

    uint16_t garb_row = ((1 << BOARD_COLS) - 1) | GARB_ROW_MARKER;

    // Shift existing rows up by 'rows' positions
    for (int r = 0; r < board_height - rows; r++) {
        board[r] = board[r + rows];
    }
    // Fill bottom 'rows' with garbage
    for (int r = board_height - rows; r < board_height; r++) {
        board[r] = garb_row;
    }
    return rows;
}

// Attack calculation — exact replica of Scorer.py
typedef struct {
    float attack;
    int new_b2b;
    int new_combo;
    bool b2b_broken;
    bool b2b_maintaining;  // true if this clear kept/started b2b (spin/tetris/PC)
    float surge;
} AttackResult;

static AttackResult compute_attack(int clears, int spin_type, int b2b, int combo,
                                   bool perfect_clear) {
    AttackResult res;
    res.attack = 0;
    res.new_b2b = b2b;
    res.new_combo = combo;
    res.b2b_broken = false;
    res.b2b_maintaining = false;
    res.surge = 0;

    if (clears > 0) {
        // B2B tracking
        if (spin_type != SPIN_NONE || clears == 4 || perfect_clear) {
            res.new_b2b = b2b + 1;
            res.b2b_maintaining = true;
        } else {
            // Breaking b2b
            if (b2b >= 4) {
                res.surge = (float)(b2b);
            }
            res.new_b2b = -1;
            if (b2b >= 0) res.b2b_broken = true;
        }

        res.new_combo = combo + 1;

        // Base attack
        if (perfect_clear) {
            int pc_table[5] = {0, 5, 6, 7, 9};
            res.attack += pc_table[clears < 5 ? clears : 4];
        } else if (spin_type == SPIN_T_FULL) {
            int ts_table[5] = {0, 2, 4, 6, 0};
            res.attack += ts_table[clears < 5 ? clears : 4];
        } else if (spin_type == SPIN_T_MINI) {
            int tm_table[5] = {0, 0, 1, 2, 0};
            res.attack += tm_table[clears < 5 ? clears : 4];
        } else {
            int no_table[5] = {0, 0, 1, 2, 4};
            res.attack += no_table[clears < 5 ? clears : 4];
        }

        // B2B bonus (applied if b2b was > -1 BEFORE this clear)
        if (b2b > -1) {
            res.attack += 1;
        }

        // Combo multiplier
        if (combo > 0) {
            if (res.attack > 0) {
                res.attack = floorf(res.attack * (1.0f + 0.25f * combo));
            } else {
                res.attack = floorf(logf(1.0f + 1.25f * combo));
            }
        }

        // Surge
        res.attack += res.surge;
    } else {
        res.new_combo = -1;
    }

    return res;
}

// ============================================================
// Internal Placement Finder (simplified BFS)
// Returns all unique (rot, col, landing_row, spin) placements.
// For depth 0, also stores BFS state for key sequence reconstruction.
// ============================================================

static int find_placements(const uint16_t* board_rows, int board_height,
                           int piece_type, Placement* out, int max_out,
                           BFSStateMeta* meta_out) {
    // meta_out may be NULL if we don't need sequence reconstruction (depth > 0)
    static BFSStateMeta meta[BFS_STATE_SPACE];
    static bool visited[BFS_STATE_SPACE];
    static int queue[BFS_QUEUE_CAPACITY];

    for (int i = 0; i < BFS_STATE_SPACE; i++) {
        visited[i] = false;
        meta[i].parent = -1;
    }

    // Spawn position: row=0, col=3, rot=0
    int start_r = 0, start_c = 3, start_rot = 0;
    int start_state = b2b_encode_state(start_r, start_c, start_rot, piece_type);

    if (start_state == -1 ||
        b2b_check_collision(board_rows, board_height, piece_type, start_rot, start_r, start_c)) {
        return 0; // Can't spawn — board is topped out
    }

    int head = 0, tail = 0;
    queue[tail++] = start_state;
    visited[start_state] = true;
    meta[start_state].depth = 0;
    meta[start_state].last_move = KEY_START;
    meta[start_state].delta_r = 0;
    meta[start_state].delta_row = 0;
    meta[start_state].delta_col = 0;

    int num_placements = 0;
    int visible_start = board_height - VISIBLE_ROWS;

    while (head != tail) {
        int curr_state = queue[head++];
        head %= BFS_QUEUE_CAPACITY;

        int r, c, rot;
        b2b_decode_state(curr_state, &r, &c, &rot, piece_type);
        int depth = meta[curr_state].depth;

        // Check landing from this position
        int land_r = b2b_hard_drop_row(board_rows, board_height, piece_type, rot, r, c);

        if (land_r >= visible_start && num_placements < max_out) {
            int spin = SPIN_NONE;
            int delta_r_val = meta[curr_state].delta_r;
            int dloc_sum = abs(meta[curr_state].delta_row) + abs(meta[curr_state].delta_col);

            if (delta_r_val != 0) {
                if (piece_type == PIECE_T) {
                    spin = detect_t_spin(board_rows, board_height, land_r, c, rot, dloc_sum);
                } else {
                    if (b2b_check_immobility(board_rows, board_height, piece_type, rot, land_r, c)) {
                        spin = SPIN_ALL_MINI;
                    }
                }
            }

            // Check for duplicate placement (same rot, col, landing_row, spin)
            bool dup = false;
            for (int i = 0; i < num_placements; i++) {
                if (out[i].rot == rot && out[i].col == c &&
                    out[i].landing_row == land_r && out[i].spin_type == spin) {
                    dup = true;
                    break;
                }
            }

            if (!dup) {
                out[num_placements].rot = rot;
                out[num_placements].col = c;
                out[num_placements].landing_row = land_r;
                out[num_placements].spin_type = spin;
                out[num_placements].delta_r = delta_r_val;
                out[num_placements].delta_loc_sum = dloc_sum;
                out[num_placements].bfs_state = curr_state;
                num_placements++;
            }
        }

        // BFS depth limit — keep paths short enough for max_len=15 sequences
        // START + (opt HOLD) + path + HARD_DROP <= 15, so path <= 12
        if (depth >= 12) continue;

        // Try all 8 moves
        int moves[] = {KEY_TAP_LEFT, KEY_TAP_RIGHT, KEY_DAS_LEFT, KEY_DAS_RIGHT,
                       KEY_CLOCKWISE, KEY_ANTICLOCKWISE, KEY_ROTATE_180, KEY_SOFT_DROP};

        for (int m = 0; m < 8; m++) {
            int key = moves[m];
            int nr = r, nc = c, nrot = rot;
            int dr = 0, drow = 0, dcol = 0;
            bool valid = false;

            if (key == KEY_TAP_LEFT) {
                if (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, c - 1)) {
                    nc--; valid = true; dcol = -1;
                }
            } else if (key == KEY_TAP_RIGHT) {
                if (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, c + 1)) {
                    nc++; valid = true; dcol = 1;
                }
            } else if (key == KEY_DAS_LEFT) {
                int tmp = c;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, tmp - 1)) tmp--;
                if (tmp != c) { nc = tmp; valid = true; dcol = nc - c; }
            } else if (key == KEY_DAS_RIGHT) {
                int tmp = c;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, tmp + 1)) tmp++;
                if (tmp != c) { nc = tmp; valid = true; dcol = nc - c; }
            } else if (key == KEY_SOFT_DROP) {
                int tmp = r;
                int max_row = B2B_PIECES[piece_type].orientations[rot].max_row;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, tmp + 1, c)) {
                    tmp++;
                    if (tmp + max_row >= board_height - 1) break;
                }
                if (tmp != r) { nr = tmp; valid = true; drow = nr - r; }
            } else {
                // Rotation
                int delta = 0;
                if (key == KEY_CLOCKWISE) delta = 1;
                else if (key == KEY_ANTICLOCKWISE) delta = 3;
                else delta = 2; // ROTATE_180

                int next_rot = (rot + delta) % 4;

                if (!b2b_check_collision(board_rows, board_height, piece_type, next_rot, r, c)) {
                    nrot = next_rot; valid = true;
                    dr = (delta == 3) ? -1 : delta;
                } else {
                    int8_t (*table)[2];
                    if (piece_type == PIECE_I) table = B2B_I_KICKS[rot][next_rot];
                    else table = B2B_KICKS[rot][next_rot];

                    int count = (key == KEY_ROTATE_180) ? 5 : 4;

                    for (int k = 0; k < count; k++) {
                        int kdr = table[k][0];
                        int kdc = table[k][1];
                        if (kdr == 0 && kdc == 0 && count == 5) continue;

                        if (!b2b_check_collision(board_rows, board_height, piece_type,
                                                 next_rot, r + kdr, c + kdc)) {
                            nr = r + kdr; nc = c + kdc; nrot = next_rot;
                            valid = true;
                            dr = (delta == 3) ? -1 : delta;
                            drow = kdr; dcol = kdc;
                            break;
                        }
                    }
                }
            }

            if (valid) {
                int next_s = b2b_encode_state(nr, nc, nrot, piece_type);
                if (next_s != -1 && !visited[next_s]) {
                    visited[next_s] = true;
                    meta[next_s].parent = curr_state;
                    meta[next_s].last_move = key;
                    meta[next_s].depth = depth + 1;
                    meta[next_s].delta_r = dr;
                    meta[next_s].delta_row = drow;
                    meta[next_s].delta_col = dcol;

                    queue[tail++] = next_s;
                    tail %= BFS_QUEUE_CAPACITY;
                }
            }
        }
    }

    // Copy BFS meta if caller wants it (for key sequence reconstruction)
    if (meta_out != NULL) {
        memcpy(meta_out, meta, sizeof(meta));
    }

    return num_placements;
}

// ============================================================
// Heuristic Evaluation
// ============================================================

// Flood-fill reachability from top of board.
// Fills reachable[] with bitmasks indicating which empty cells are reachable
// from the top row via orthogonal movement through empty cells.
static void compute_reachability(const uint16_t* board, int board_height,
                                  uint16_t* reachable) {
    memset(reachable, 0, sizeof(uint16_t) * board_height);

    static int flood_queue[BOARD_ROWS * BOARD_COLS * 2]; // row, col pairs
    int fh = 0, ft = 0;

    for (int c = 0; c < BOARD_COLS; c++) {
        if (!(board[0] & (1 << c))) {
            reachable[0] |= (1 << c);
            flood_queue[ft++] = 0;
            flood_queue[ft++] = c;
        }
    }

    while (fh < ft) {
        int r = flood_queue[fh++];
        int c = flood_queue[fh++];

        int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int d = 0; d < 4; d++) {
            int nr = r + dirs[d][0];
            int nc = c + dirs[d][1];
            if (nr < 0 || nr >= board_height || nc < 0 || nc >= BOARD_COLS) continue;
            uint16_t bit = (1 << nc);
            if ((board[nr] & bit) || (reachable[nr] & bit)) continue;
            reachable[nr] |= bit;
            flood_queue[ft++] = nr;
            flood_queue[ft++] = nc;
        }
    }
}

// Count enclosed holes using precomputed reachability.
// Holes that are part of a spin setup (immobile_cells) are excluded —
// they're intentional cavities, not structural damage.
static int count_enclosed_holes_from_reachability(const uint16_t* board,
                                                   int board_height,
                                                   const uint16_t* reachable,
                                                   const uint16_t* immobile_cells) {
    uint16_t full_mask = (1 << BOARD_COLS) - 1;
    int holes = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t empty = (~board[r]) & full_mask;
        uint16_t enclosed = empty & (~reachable[r]);
        enclosed &= ~immobile_cells[r];  // exempt spin setup cells
        uint16_t v = enclosed;
        while (v) { holes++; v &= v - 1; }
    }
    return holes;
}

// Compute "hole ceiling weight": for each enclosed hole, count the number
// of filled cells above it in the same column, weighted by how high the
// hole is in the stack (higher holes = more urgent = higher weight).
//
// This penalizes upstacking over enclosed holes — each filled cell placed
// above an enclosed hole makes it harder to clear, and holes near the top
// of the stack are more dangerous.
//
// Returns a float score (not an integer count) because of height weighting.
static float compute_hole_ceiling_weight(const uint16_t* board, int board_height,
                                          const uint16_t* reachable,
                                          const uint16_t* immobile_cells) {
    float total_weight = 0.0f;

    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);

        // Scan column top-to-bottom, track filled cells above
        int filled_above = 0;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                filled_above++;
            } else {
                // Empty cell — check if it's an enclosed hole
                bool enclosed = !(reachable[r] & bit);
                bool is_setup = (immobile_cells[r] & bit) != 0;
                if (enclosed && !is_setup && filled_above > 0) {
                    int hole_height = board_height - r;
                    float height_factor = (float)hole_height / (float)board_height;
                    total_weight += (float)filled_above * (1.0f + height_factor);
                }
            }
        }
    }

    return total_weight;
}

// ============================================================
// Spin Setup Detection
// ============================================================

// Check if a cell is filled (or out of bounds = filled)
static inline bool cell_filled(const uint16_t* board, int board_height, int r, int c) {
    if (r < 0 || r >= board_height || c < 0 || c >= BOARD_COLS) return true;
    return (board[r] & (1 << c)) != 0;
}

static inline bool cell_empty(const uint16_t* board, int board_height, int r, int c) {
    if (r < 0 || r >= board_height || c < 0 || c >= BOARD_COLS) return false;
    return (board[r] & (1 << c)) == 0;
}

// Detect T-spin setups: look for T-shaped slots where a T piece could spin in.
// A T-slot is a pattern where:
//   - There's a T-shaped cavity (3 empty cells in T formation)
//   - At least 3 of the 4 corners around the T center are filled
//   - The T piece could actually reach and spin into this slot
//
// We check for all 4 T-spin orientations at each board position.
// Returns: number of T-spin setups found (0, 1, or more)
// t_slot_quality: set to best quality found (0=none, 1=mini possible, 2=full T-spin)
static int detect_t_spin_setups(const uint16_t* board, int board_height,
                                int* t_slot_quality) {
    int setups = 0;
    int best_quality = 0;

    // For each possible T-piece center position (row, col),
    // check if a T-spin could happen in each of the 4 rotations.
    //
    // T piece orientations (center at [1,1] in 3x3 grid):
    // Rot 0: [0,1] [1,0] [1,1] [1,2] -> points up, spins from above
    // Rot 1: [0,1] [1,1] [1,2] [2,1] -> points right, spins from right
    // Rot 2: [1,0] [1,1] [1,2] [2,1] -> points down, spins from below
    // Rot 3: [0,1] [1,0] [1,1] [2,1] -> points left, spins from left

    // We look for the T-slot pattern: the 3 cells of the T (excluding center-back)
    // must be empty, and at least 3 of 4 corners must be filled.

    for (int r = 0; r < board_height - 2; r++) {
        for (int c = 0; c < BOARD_COLS - 2; c++) {
            // Corner positions (in the 3x3 grid anchored at r,c)
            bool tl = cell_filled(board, board_height, r, c);
            bool tr = cell_filled(board, board_height, r, c + 2);
            bool bl = cell_filled(board, board_height, r + 2, c);
            bool br = cell_filled(board, board_height, r + 2, c + 2);
            int corner_count = tl + tr + bl + br;

            if (corner_count < 3) continue;

            // Check each T rotation for a valid slot

            // Rot 2 (T points down, most common T-spin: overhang from above)
            // T cells: [1,0] [1,1] [1,2] [2,1] — center is [1,1]
            // Need these cells empty:
            if (cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                // Front corners for rot 2: BL and BR
                int front = bl + br;
                int back = tl + tr;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2; // Full T-spin
                else if (front == 1 && back == 2) quality = 1; // Mini possible
                if (quality > 0) {
                    // Check that the slot is accessible: the entry point [0,1] should
                    // have some path from above (at least the cell above should be empty)
                    if (cell_empty(board, board_height, r, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                    }
                }
            }

            // Rot 0 (T points up — less common, needs slot below)
            // T cells: [0,1] [1,0] [1,1] [1,2] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2)) {
                int front = tl + tr;
                int back = bl + br;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    // Check accessibility from above
                    if (cell_empty(board, board_height, r, c) ||
                        cell_empty(board, board_height, r, c + 2)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                    }
                }
            }

            // Rot 1 (T points right)
            // T cells: [0,1] [1,1] [1,2] [2,1] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                int front = tr + br;
                int back = tl + bl;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    if (cell_empty(board, board_height, r, c + 2) ||
                        r == 0 ||
                        cell_empty(board, board_height, r - 1, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                    }
                }
            }

            // Rot 3 (T points left)
            // T cells: [0,1] [1,0] [1,1] [2,1] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                int front = tl + bl;
                int back = tr + br;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    if (cell_empty(board, board_height, r, c) ||
                        r == 0 ||
                        cell_empty(board, board_height, r - 1, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                    }
                }
            }
        }
    }

    *t_slot_quality = best_quality;
    return setups;
}

// ============================================================
// Spin-Placement Counting (immobile placements)
//
// For each piece type (excluding O and T) × each orientation × each
// board position, check:
//   1. FITS:      All piece cells are empty on the board
//   2. REACHABLE: At least one piece cell is reachable from surface
//   3. IMMOBILE:  Piece cannot move in any cardinal direction
//                 (the actual ALL_MINI criterion — matches
//                 b2b_check_immobility used during placement)
//
// O is excluded (can't spin). T is excluded (T-spins use the
// 3-corner rule, handled separately by detect_t_spin_setups).
//
// Uses the existing B2B_PIECES definitions (row_masks bitmask format).
// Pieces are indexed 1-7 (PIECE_I through PIECE_Z), skipping PIECE_N=0.
// ============================================================

typedef struct {
    float weighted_immobile;            // Queue-weighted sum of truly immobile placements
    float weighted_immobile_clearing;   // Queue-weighted sum of immobile + line-clearing placements
    float weighted_immobile_lines;      // Queue-weighted sum of clearable lines from immobile placements
} ImmobilePlacementResult;

// count_immobile_placements scans only pieces that appear in the upcoming queue
// (hold piece + next queue pieces). Each piece's contribution is weighted by
// how soon it appears: the next piece gets weight 1.0, the one after gets 0.5,
// then 0.33, etc. (1/position). This ensures the heuristic rewards setups
// that can be resolved SOON rather than speculative cavities for distant pieces.
//
// Populates two per-row bitmasks:
//   immobile_cells[]   — cells in ANY valid immobile placement (used for
//                         wasted-hole detection: reachable holes not in here
//                         are "wasted").
//   clearing_cells[]   — cells in immobile placements that CLEAR at least one
//                         line (used to exempt productive spin-setup holes from
//                         hole penalties — only clearing setups earn exemption).
//
// upcoming_pieces: array of piece types (1-7) to check, ordered by priority
// num_upcoming: length of upcoming_pieces
static ImmobilePlacementResult count_immobile_placements(
    const uint16_t* board, int board_height,
    const uint16_t* reachable,
    uint16_t* immobile_cells,        // output: ALL immobile placement cells
    uint16_t* clearing_cells,        // output: only clearing immobile cells
    const int* upcoming_pieces,
    int num_upcoming
) {
    ImmobilePlacementResult res = {0.0f, 0.0f, 0.0f};
    memset(immobile_cells, 0, sizeof(uint16_t) * board_height);
    memset(clearing_cells, 0, sizeof(uint16_t) * board_height);

    if (num_upcoming <= 0) return res;

    uint16_t full_mask = (1 << BOARD_COLS) - 1;

    // Find highest filled row to bound the scan
    int top_filled = board_height;
    for (int r = 0; r < board_height; r++) {
        if (board[r] != 0) { top_filled = r; break; }
    }
    // Start a few rows above to catch pieces partially above the stack
    int scan_start = top_filled - 3;
    if (scan_start < 0) scan_start = 0;

    // Build per-piece-type best weight from queue position.
    // If a piece appears multiple times in the queue, use the best (earliest)
    // weight. Weight for position i = 1.0 / (i + 1).
    float piece_weight[8]; // indexed by piece type (0=N unused, 1-7)
    memset(piece_weight, 0, sizeof(piece_weight));
    for (int i = 0; i < num_upcoming; i++) {
        int pt = upcoming_pieces[i];
        if (pt < PIECE_I || pt > PIECE_Z) continue;
        float w = 1.0f / (float)(i + 1);
        if (w > piece_weight[pt]) {
            piece_weight[pt] = w;
        }
    }

    // Iterate over piece types that have non-zero weight
    for (int pt = PIECE_I; pt <= PIECE_Z; pt++) {
        if (piece_weight[pt] <= 0.0f) continue;
        // Skip O piece (can't spin) and T piece (uses corner rule, not immobility)
        if (pt == PIECE_O || pt == PIECE_T) continue;

        float w = piece_weight[pt];

        for (int rot = 0; rot < ROTATIONS; rot++) {
            PieceOrientation* ori = &B2B_PIECES[pt].orientations[rot];

            for (int r = scan_start; r < board_height; r++) {
                if (r + ori->min_row < 0) continue;
                if (r + ori->max_row >= board_height) break;

                for (int c = -ori->min_col; c < BOARD_COLS - ori->max_col; c++) {
                    // 1. FITS: all piece cells are empty
                    bool fits = true;
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        if (board[br] & shifted) { fits = false; break; }
                    }
                    if (!fits) continue;

                    // 2. REACHABLE: at least one piece cell is reachable
                    bool any_reachable = false;
                    for (int i = 0; i < 4 && !any_reachable; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        if (reachable[br] & shifted) any_reachable = true;
                    }
                    if (!any_reachable) continue;

                    // 3. TRULY IMMOBILE: piece cannot move in any cardinal
                    //    direction (ALL_MINI criterion). Bail on first
                    //    unblocked direction.
                    if (!b2b_check_collision(board, board_height, pt, rot, r + 1, c)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r - 1, c)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r, c + 1)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r, c - 1)) continue;

                    // Valid immobile placement! Count clearable lines first.
                    int lines = 0;
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        uint16_t combined = board[br] | shifted;
                        if ((combined & full_mask) == full_mask) lines++;
                    }

                    // Mark cells: ALL placements go into immobile_cells
                    // (for wasted-hole detection); only CLEARING placements
                    // go into clearing_cells (for hole-penalty exemption).
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        immobile_cells[br] |= shifted;
                        if (lines > 0) clearing_cells[br] |= shifted;
                    }

                    res.weighted_immobile += w;
                    if (lines > 0) {
                        res.weighted_immobile_clearing += w;
                        res.weighted_immobile_lines += w * (float)lines;
                    }
                }
            }
        }
    }

    return res;
}

// Count how many holes are "deep" (have 2+ filled cells above them in the same column).
// Holes that are part of a spin setup (immobile_cells) are excluded.
static int count_deep_holes(const uint16_t* board, int board_height,
                            const uint16_t* immobile_cells) {
    int deep = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        int filled_above = 0;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                filled_above++;
            } else if (filled_above >= 2) {
                if (!(immobile_cells[r] & bit)) {
                    deep++;
                }
            }
        }
    }
    return deep;
}

// Compute per-column heights and derived stats
typedef struct {
    int col_heights[BOARD_COLS];
    int max_height;
    float avg_height;
    float bumpiness;
    int holes;
    int hole_columns;       // Number of distinct columns containing holes
    int clearable_rows;     // Rows with >= 8 filled cells
    int almost_full_rows;   // Rows with exactly 9 filled cells
    int well_depth;         // Depth of the deepest single well
    int well_count;         // Number of wells (columns lower than both neighbors by 2+)
    int accessible_9_rows;  // Almost-full rows where the hole has clear path from above
    int blocked_9_rows;     // Almost-full rows where the hole is blocked
    int well_col;           // Column of the deepest well (-1 if no well)
    int well_aligned_9;     // Accessible 9-rows where the gap is in the well column
    int non_well_9;         // Accessible 9-rows where the gap is NOT in the well column
    int t_spin_setups;      // Number of T-spin setups detected
    int t_slot_quality;     // Best T-slot quality (0=none, 1=mini, 2=full)
    int deep_holes;         // Holes buried under 2+ filled cells
    int edge_well_depth;    // Deepest well in columns 0, 1, 8, or 9
    float hole_ceiling_weight; // Weighted count of filled cells above enclosed holes
    float immobile_placements;           // Queue-weighted truly-immobile spin-placement count
    float immobile_clearing_placements;  // Queue-weighted immobile + line-clearing placements
    float immobile_clearable_lines;      // Queue-weighted sum of clearable lines from immobile placements
    int wasted_holes;                    // Non-enclosed holes not part of any immobile placement
} BoardStats;

static BoardStats compute_board_stats(const uint16_t* board, int board_height,
                                      const int* upcoming_pieces, int num_upcoming) {
    BoardStats s;
    memset(&s, 0, sizeof(s));

    uint16_t full_mask = (1 << BOARD_COLS) - 1;

    // Column heights
    for (int c = 0; c < BOARD_COLS; c++) {
        s.col_heights[c] = 0;
        uint16_t bit = (1 << c);
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                s.col_heights[c] = board_height - r;
                break;
            }
        }
    }

    // Max and average height
    float total_h = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if (s.col_heights[c] > s.max_height) s.max_height = s.col_heights[c];
        total_h += s.col_heights[c];
    }
    s.avg_height = total_h / BOARD_COLS;

    // Bumpiness: sum of absolute column height differences
    for (int c = 0; c < BOARD_COLS - 1; c++) {
        s.bumpiness += fabsf((float)(s.col_heights[c] - s.col_heights[c + 1]));
    }

    // Flood-fill reachability (used for holes and spin placements)
    uint16_t reachable[BOARD_ROWS];
    compute_reachability(board, board_height, reachable);

    // ── Immobile spin-placement counting FIRST ──────────────────
    // Computed before hole metrics so that immobile_cells[] is available
    // to exempt spin-setup holes from all hole penalties.  Holes that
    // are part of a valid immobile placement are intentional cavities,
    // not structural damage.
    uint16_t immobile_cells[BOARD_ROWS];
    uint16_t clearing_cells[BOARD_ROWS];
    ImmobilePlacementResult ipr = count_immobile_placements(board, board_height, reachable,
                                                             immobile_cells, clearing_cells,
                                                             upcoming_pieces, num_upcoming);
    s.immobile_placements = ipr.weighted_immobile;
    s.immobile_clearing_placements = ipr.weighted_immobile_clearing;
    s.immobile_clearable_lines = ipr.weighted_immobile_lines;

    // ── Hole metrics (setup-aware) ──────────────────────────────
    // Hole penalties exclude cells in clearing_cells[] — those are
    // intentional cavities for spin setups that would clear lines
    // (b2b-maintaining downstack).  Non-clearing immobile placements
    // are NOT exempted: holes without clearing potential are still
    // structural damage.

    // ── Hole metrics ──────────────────────────────────────────
    // Holes are penalized uniformly — spin-setup holes are NOT exempted
    // here.  Instead, the spin-setup REWARDS in evaluate_state are strong
    // enough to outweigh the hole cost, so the bot only creates holes
    // when it has a productive clearing setup, not speculatively.
    uint16_t no_exempt[BOARD_ROWS];
    memset(no_exempt, 0, sizeof(no_exempt));

    s.holes = count_enclosed_holes_from_reachability(board, board_height, reachable, no_exempt);

    s.hole_columns = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        bool found_filled = false;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                found_filled = true;
            } else if (found_filled) {
                s.hole_columns++;
                break;
            }
        }
    }

    s.deep_holes = count_deep_holes(board, board_height, no_exempt);
    s.hole_ceiling_weight = compute_hole_ceiling_weight(board, board_height, reachable, no_exempt);

    // T-spin setup detection
    s.t_spin_setups = detect_t_spin_setups(board, board_height, &s.t_slot_quality);

    // Nearly-complete rows (8+ out of 10 cells filled)
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r] & full_mask;
        int bits = 0;
        uint16_t v = row;
        while (v) { bits++; v &= v - 1; }
        if (bits >= 8) s.clearable_rows++;
        if (bits == 9) s.almost_full_rows++;
    }

    // Well detection: columns lower than both neighbors by 2+
    s.well_depth = 0;
    s.well_count = 0;
    s.well_col = -1;
    s.edge_well_depth = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        int left_h = (c > 0) ? s.col_heights[c - 1] : board_height;
        int right_h = (c < BOARD_COLS - 1) ? s.col_heights[c + 1] : board_height;
        int min_neighbor = (left_h < right_h) ? left_h : right_h;
        int depth = min_neighbor - s.col_heights[c];
        if (depth >= 2) {
            s.well_count++;
            if (depth > s.well_depth) {
                s.well_depth = depth;
                s.well_col = c;
            }
            if (c <= 1 || c >= 8) {
                if (depth > s.edge_well_depth) s.edge_well_depth = depth;
            }
        }
    }

    // Accessible vs blocked almost-full rows
    s.accessible_9_rows = 0;
    s.blocked_9_rows = 0;
    s.well_aligned_9 = 0;
    s.non_well_9 = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r] & full_mask;
        int bits = 0;
        uint16_t v = row;
        while (v) { bits++; v &= v - 1; }
        if (bits != 9) continue;

        uint16_t hole_mask = (~row) & full_mask;
        int hole_col = -1;
        for (int c = 0; c < BOARD_COLS; c++) {
            if (hole_mask & (1 << c)) { hole_col = c; break; }
        }
        if (hole_col < 0) continue;

        bool accessible = true;
        uint16_t hbit = (1 << hole_col);
        for (int rr = r - 1; rr >= 0; rr--) {
            if (board[rr] & hbit) { accessible = false; break; }
        }
        if (accessible) {
            s.accessible_9_rows++;
            if (s.well_col >= 0 && hole_col == s.well_col) {
                s.well_aligned_9++;
            } else {
                s.non_well_9++;
            }
        } else {
            s.blocked_9_rows++;
        }
    }

    // Wasted holes: non-enclosed holes (reachable empty cells below a filled
    // cell in the same column) that are NOT part of any immobile placement.
    s.wasted_holes = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        bool found_filled = false;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                found_filled = true;
            } else if (found_filled) {
                bool is_reachable = (reachable[r] & bit) != 0;
                bool is_immobile = (immobile_cells[r] & bit) != 0;
                if (is_reachable && !is_immobile) {
                    s.wasted_holes++;
                }
            }
        }
    }

    return s;
}

static float evaluate_state(const SearchState* state, int board_height,
                            const int* queue, int queue_len) {
    float score = 0.0f;

    // Build the upcoming pieces list: hold piece (if any) + remaining queue
    int upcoming[MAX_SEARCH_DEPTH + 2];
    int num_upcoming = 0;
    if (state->hold_piece != PIECE_N) {
        upcoming[num_upcoming++] = state->hold_piece;
    }
    for (int i = state->next_queue_idx; i < queue_len && num_upcoming < MAX_SEARCH_DEPTH + 2; i++) {
        upcoming[num_upcoming++] = queue[i];
    }

    BoardStats bs = compute_board_stats(state->board, board_height, upcoming, num_upcoming);

    int max_allowed = board_height - 4; // rows 0-3 are invisible/death zone

    // Effective height: stack height + remaining garbage that hasn't
    // been pushed onto the board yet.  Garbage that HAS been pushed is
    // already reflected in bs.max_height (those rows are on the board
    // with GARB_ROW_MARKER), so we only add the un-pushed remainder.
    int effective_h = bs.max_height + state->garbage_remaining;

    // Instant death
    if (effective_h >= max_allowed) {
        return -1e6f;
    }

    float h_ratio = (float)effective_h / (float)max_allowed;

    // ── PRIORITY 1: SURVIVAL ──────────────────────────────────

    // Height penalty — quadratic + cubic for strong high-board aversion
    score -= W_HEIGHT * 15.0f * h_ratio * h_ratio;
    score -= W_HEIGHT * 15.0f * h_ratio * h_ratio * h_ratio;

    // Garbage cancellation reward: ATTACK (not lines cleared) is what
    // actually removes garbage from the queue in-game.  When garbage is
    // pending, cancelling it is the single most important thing — this
    // reward is intentionally strong enough to override b2b maintenance
    // and board-shape preferences.
    if (state->garbage_remaining > 0 && state->total_attack > 0.0f) {
        float cancellable = fminf(state->total_attack, (float)state->garbage_remaining);
        score += W_GARB_CANCEL * cancellable;
    }

    // Average height penalty
    score -= W_AVG_HEIGHT * bs.avg_height;

    // Bumpiness penalty
    score -= W_BUMPINESS * bs.bumpiness;

    // ── HOLES: nuanced penalty ────────────────────────────────
    // Use diminishing returns: first hole is bad, but each additional hole
    // is less incrementally bad (the damage is already done).
    // Deep holes (buried under 2+ cells) are worse than shallow ones.
    // Holes that are part of a spin setup are partially forgiven.

    float hole_mult = 1.0f + 0.5f * h_ratio;  // less height scaling than before

    // Base hole penalty with diminishing returns: sqrt-based
    if (bs.holes > 0) {
        // First hole costs W_HOLES, each additional costs less
        // Total: W_HOLES * (1 + 0.7 + 0.58 + 0.5 + ...) via sqrt scaling
        float hole_penalty = W_HOLES * sqrtf((float)bs.holes) * hole_mult;
        score -= hole_penalty;
    }

    // Extra penalty for deep holes (these are truly hard to fix)
    if (bs.deep_holes > 0) {
        score -= W_DEEP_HOLES * (float)bs.deep_holes * hole_mult;
    }

    // Penalty for filled cells above enclosed holes (discourages upstacking
    // over holes). The weight already accounts for hole height — holes higher
    // in the stack contribute more penalty per filled cell above them.
    if (bs.hole_ceiling_weight > 0.0f) {
        score -= W_HOLE_CEILING * bs.hole_ceiling_weight;
    }

    // Penalty for non-enclosed holes that aren't part of any tucked placement.
    // These are reachable gaps under the stack surface that serve no spin-setup
    // purpose — they're pure structural damage that wastes board space.
    if (bs.wasted_holes > 0) {
        score -= W_WASTED_HOLE * (float)bs.wasted_holes * hole_mult;
    }

    // Penalty for distinct columns with holes (spread-out holes are worse)
    score -= W_HOLE_COL * (float)bs.hole_columns * hole_mult;

    // Hole forgiveness when clearing spin setups are present:
    // holes tied to immobile-clearing placements or T-spin setups are
    // partially forgiven — they're structural, not damage.
    if (bs.holes > 0) {
        float setup_count = bs.immobile_clearing_placements + (float)bs.t_spin_setups;
        if (setup_count > 0.0f) {
            float forgiveness = 1.5f * fminf(setup_count, (float)bs.holes);
            score += forgiveness * hole_mult;
        }
    }

    // Urgency factor used by combo, spin-clear, and other height-sensitive rewards
    float urgency = 1.0f + 2.0f * h_ratio;

    // Well bonus: having exactly 1 well (for Tetris I-piece) is good for B2B
    if (bs.well_count == 1 && bs.well_depth >= 4 && bs.well_depth <= 8) {
        score += 3.0f;
    } else if (bs.well_count == 1 && bs.well_depth >= 2 && bs.well_depth <= 3) {
        score += 1.0f; // Small well is still somewhat useful
    }
    if (bs.well_count > 1) {
        score -= 1.5f * (bs.well_count - 1);
    }

    // Edge-column well penalty: wells in columns 0, 1, 8, 9 are hard to
    // clear efficiently (only I-piece vertical or awkward placements reach
    // them) and tend to create garbage-vulnerable board shapes.
    if (bs.edge_well_depth >= 3) {
        score -= 2.0f * (float)(bs.edge_well_depth - 2);
    }

    // ── PRIORITY 2: B2B MAINTENANCE & SPIN SETUPS ─────────────

    // Emergency factor: when board is very high, B2B is less important
    // BUT don't drop it too far — spin clears are still the most efficient
    // way to clear lines at any height.
    float b2b_scale = 1.0f;
    if (h_ratio > 0.6f) {
        b2b_scale = 1.0f - (h_ratio - 0.6f) * 1.0f;
        if (b2b_scale < 0.35f) b2b_scale = 0.35f;
    }

    // Flat bonus for having B2B active
    if (state->b2b >= 0) {
        score += 10.0f * b2b_scale;
    }

    // Additional value for higher B2B levels (logarithmic — diminishing returns)
    float b2b_val = (state->b2b > 0) ? (float)state->b2b : 0.0f;
    float combo_val = (state->combo > 0) ? (float)state->combo : 0.0f;
    score += W_B2B * logf(2.0f + b2b_val) * b2b_scale;
    // Combo reward scaled by urgency: at low boards combos are nice; at
    // high effective height they become critical for survival because each
    // consecutive clear prevents garbage from pushing and reduces the stack.
    score += W_COMBO * combo_val * combo_val * urgency;

    // B2B break penalty — only allow breaking in the danger zone.
    //
    // Breaking b2b triggers a surge attack (real game mechanic) that
    // makes the break look attractive through W_ATTACK / W_SPIKE.
    // To prevent the bot from farming b2b to 4 just to break for surge,
    // the penalty is prohibitively steep at safe board heights and only
    // relaxes once effective height enters the danger zone.
    //
    // The reward for *building* b2b (log bonus, flat bonus) remains
    // positive at all heights, so building is always incentivized.
    if (state->b2b_broken && state->prev_b2b >= 0) {
        float break_cost = W_B2B_BREAK;
        if (state->prev_b2b <= 1) {
            break_cost *= 0.5f;
        } else {
            break_cost *= (1.0f + 0.3f * logf(1.0f + (float)state->prev_b2b));
        }
        // Inverse urgency: when downstacking pressure is high (urgency
        // large), breaking is cheap.  When the board is safe (urgency ≈ 1),
        // breaking is 3× base cost.  Same urgency variable drives both
        // combo/spin-clear scaling and break tolerance.
        score -= break_cost * (3.0f / urgency);
    }

    // ── SPIN SETUP REWARDS ────────────────────────────────────
    // Reward boards that have recognizable spin setups, especially T-spin slots.
    // This encourages the algorithm to BUILD spin setups rather than just
    // accidentally finding them.

    if (bs.t_spin_setups > 0) {
        // Quality 2 = full T-spin possible, quality 1 = mini
        float t_reward = 0.0f;
        if (bs.t_slot_quality == 2) {
            t_reward = W_TSLOT;
        } else if (bs.t_slot_quality == 1) {
            t_reward = W_TSLOT * 0.4f;
        }
        // Diminishing returns for multiple setups
        t_reward *= (1.0f + 0.3f * fminf((float)(bs.t_spin_setups - 1), 2.0f));
        score += t_reward * b2b_scale;
    }

    // Immobile spin placements: reward boards where upcoming queue pieces
    // could be placed in a truly immobile position (ALL_MINI eligible).
    // Values are queue-weighted so setups for the next piece score highest.
    if (bs.immobile_placements > 0.0f) {
        float imm_reward = W_SPIN_SETUP * sqrtf(bs.immobile_placements);
        score += imm_reward * b2b_scale;
    }

    // Immobile placements that would also clear lines — these represent
    // b2b-maintaining downstack opportunities (spin clears). Emphasized
    // over base immobile reward because they directly increase b2b.
    if (bs.immobile_clearing_placements > 0.0f) {
        float line_reward = W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements);
        // Bonus scales with total lines clearable (doubles > singles)
        line_reward += 1.0f * fminf(bs.immobile_clearable_lines, 8.0f);
        score += line_reward * b2b_scale;
    }

    // ── PRIORITY 3: ATTACK OUTPUT ───────────────────────────────
    // Only b2b-maintaining attack (spins, tetrises, PCs) counts toward
    // attack rewards.  Non-b2b clears still reduce height (survival) and
    // cancel garbage, but earn zero attack credit.  This naturally makes
    // the bot avoid breaking b2b — there's no upside to non-b2b clears
    // beyond height relief.

    // Spike reward: superlinear in single b2b-maintaining attack
    score += W_SPIKE * powf(state->max_b2b_attack, 1.5f);

    // B2B attack bonus — linear component
    if (state->b2b_attack > 0) {
        score += W_ATTACK * state->b2b_attack;
    }

    // Efficiency bonus: b2b attack-per-piece ratio in the search path.
    {
        int depth = state->next_queue_idx + 1;  // pieces placed so far
        if (state->b2b_attack > 0 && depth > 0) {
            float app = state->b2b_attack / (float)depth;
            // Superlinear reward for high APP — strongly favors efficient b2b paths
            score += W_APP_BONUS * app * app;
        }
    }

    // Consecutive-attack streak reward: superlinear so each additional
    // attack-sending placement is increasingly valuable. This drives
    // commitment to combos once started (especially after a b2b break
    // with surge, which seeds the streak with a large initial value).
    if (state->streak_attack > 0.0f) {
        score += W_STREAK * powf(state->streak_attack, 1.5f);
    }

    return score;
}

// ============================================================
// Score Decomposition (for heuristic influence analysis)
// ============================================================

#define NUM_DECOMPOSE 22

#define D_HEIGHT         0
#define D_GARB_CANCEL    1
#define D_AVG_HEIGHT     2
#define D_BUMPINESS      3
#define D_HOLES          4
#define D_DEEP_HOLES     5
#define D_HOLE_CEILING   6
#define D_WASTED_HOLES   7
#define D_HOLE_COLS      8
#define D_HOLE_FORGIVE   9
#define D_WELL           10
#define D_B2B_FLAT       11
#define D_B2B_LOG        12
#define D_COMBO          13
#define D_B2B_BREAK      14
#define D_TSLOT          15
#define D_IMMOBILE_SETUP 16
#define D_IMMOBILE_CLEAR 17
#define D_SPIKE          18
#define D_ATTACK         19
#define D_APP            20
#define D_STREAK         21

int b2b_get_num_decompose(void) { return NUM_DECOMPOSE; }

// Mirrors evaluate_state() exactly, but writes each term to d[] individually.
static void evaluate_state_decompose(const SearchState* state, int board_height,
                                      const int* queue, int queue_len,
                                      float* d) {
    memset(d, 0, sizeof(float) * NUM_DECOMPOSE);

    int upcoming[MAX_SEARCH_DEPTH + 2];
    int num_upcoming = 0;
    if (state->hold_piece != PIECE_N) upcoming[num_upcoming++] = state->hold_piece;
    for (int i = state->next_queue_idx; i < queue_len && num_upcoming < MAX_SEARCH_DEPTH + 2; i++)
        upcoming[num_upcoming++] = queue[i];

    BoardStats bs = compute_board_stats(state->board, board_height, upcoming, num_upcoming);

    int max_allowed = board_height - 4;
    int effective_h = bs.max_height + state->garbage_remaining;
    if (effective_h >= max_allowed) { d[D_HEIGHT] = -1e6f; return; }

    float h_ratio = (float)effective_h / (float)max_allowed;
    float hole_mult = 1.0f + 0.5f * h_ratio;
    float urgency = 1.0f + 2.0f * h_ratio;
    float b2b_scale = 1.0f;
    if (h_ratio > 0.6f) { b2b_scale = 1.0f - (h_ratio - 0.6f); if (b2b_scale < 0.35f) b2b_scale = 0.35f; }

    // SURVIVAL
    d[D_HEIGHT] = -(W_HEIGHT * 15.0f * h_ratio * h_ratio)
                  -(W_HEIGHT * 15.0f * h_ratio * h_ratio * h_ratio);
    if (state->garbage_remaining > 0 && state->total_attack > 0.0f)
        d[D_GARB_CANCEL] = W_GARB_CANCEL * fminf(state->total_attack, (float)state->garbage_remaining);
    d[D_AVG_HEIGHT] = -W_AVG_HEIGHT * bs.avg_height;
    d[D_BUMPINESS]  = -W_BUMPINESS * bs.bumpiness;

    // HOLES
    if (bs.holes > 0)
        d[D_HOLES] = -W_HOLES * sqrtf((float)bs.holes) * hole_mult;
    if (bs.deep_holes > 0)
        d[D_DEEP_HOLES] = -W_DEEP_HOLES * (float)bs.deep_holes * hole_mult;
    if (bs.hole_ceiling_weight > 0.0f)
        d[D_HOLE_CEILING] = -W_HOLE_CEILING * bs.hole_ceiling_weight;
    if (bs.wasted_holes > 0)
        d[D_WASTED_HOLES] = -W_WASTED_HOLE * (float)bs.wasted_holes * hole_mult;
    d[D_HOLE_COLS] = -W_HOLE_COL * (float)bs.hole_columns * hole_mult;
    if (bs.holes > 0) {
        float sc = bs.immobile_clearing_placements + (float)bs.t_spin_setups;
        if (sc > 0.0f) d[D_HOLE_FORGIVE] = 1.5f * fminf(sc, (float)bs.holes) * hole_mult;
    }

    // WELLS
    float well = 0.0f;
    if (bs.well_count == 1 && bs.well_depth >= 4 && bs.well_depth <= 8) well += 3.0f;
    else if (bs.well_count == 1 && bs.well_depth >= 2 && bs.well_depth <= 3) well += 1.0f;
    if (bs.well_count > 1) well -= 1.5f * (bs.well_count - 1);
    if (bs.edge_well_depth >= 3) well -= 2.0f * (float)(bs.edge_well_depth - 2);
    d[D_WELL] = well;

    // B2B / COMBO
    if (state->b2b >= 0) d[D_B2B_FLAT] = 10.0f * b2b_scale;
    float b2b_val = (state->b2b > 0) ? (float)state->b2b : 0.0f;
    d[D_B2B_LOG] = W_B2B * logf(2.0f + b2b_val) * b2b_scale;
    float combo_val = (state->combo > 0) ? (float)state->combo : 0.0f;
    d[D_COMBO] = W_COMBO * combo_val * combo_val * urgency;

    if (state->b2b_broken && state->prev_b2b >= 0) {
        float bc = W_B2B_BREAK;
        if (state->prev_b2b <= 1) bc *= 0.5f;
        else bc *= (1.0f + 0.3f * logf(1.0f + (float)state->prev_b2b));
        d[D_B2B_BREAK] = -bc * (3.0f / urgency);
    }

    // SPIN SETUPS
    if (bs.t_spin_setups > 0) {
        float tr = 0.0f;
        if (bs.t_slot_quality == 2) tr = W_TSLOT;
        else if (bs.t_slot_quality == 1) tr = W_TSLOT * 0.4f;
        tr *= (1.0f + 0.3f * fminf((float)(bs.t_spin_setups - 1), 2.0f));
        d[D_TSLOT] = tr * b2b_scale;
    }
    if (bs.immobile_placements > 0.0f)
        d[D_IMMOBILE_SETUP] = W_SPIN_SETUP * sqrtf(bs.immobile_placements) * b2b_scale;
    if (bs.immobile_clearing_placements > 0.0f) {
        float lr = W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements);
        lr += 1.0f * fminf(bs.immobile_clearable_lines, 8.0f);
        d[D_IMMOBILE_CLEAR] = lr * b2b_scale;
    }

    // ATTACK (b2b-maintaining only)
    d[D_SPIKE] = W_SPIKE * powf(state->max_b2b_attack, 1.5f);
    if (state->b2b_attack > 0) d[D_ATTACK] = W_ATTACK * state->b2b_attack;
    {
        int depth = state->next_queue_idx + 1;
        if (state->b2b_attack > 0 && depth > 0) {
            float app = state->b2b_attack / (float)depth;
            d[D_APP] = W_APP_BONUS * app * app;
        }
    }

    // STREAK
    if (state->streak_attack > 0.0f)
        d[D_STREAK] = W_STREAK * powf(state->streak_attack, 1.5f);
}

// Exported: enumerate depth-0 placements, return decomposed scores.
// decompose_out must hold max_placements * NUM_DECOMPOSE floats.
// Returns number of placements written.
int b2b_decompose_c(
    const uint16_t* board_rows, int board_height,
    int active_piece, int hold_piece,
    const int* queue, int queue_len,
    int b2b, int combo, int total_garbage,
    int garbage_push_delay,
    float* decompose_out, int max_placements
) {
    if (!b2b_initialized) b2b_init_pieces();

    int init_gt = (garbage_push_delay > 0) ? garbage_push_delay : 0;
    int count = 0;
    Placement placements[MAX_PLACEMENTS];
    int np;

    // Active piece placements
    np = find_placements(board_rows, board_height, active_piece, placements, MAX_PLACEMENTS, NULL);
    for (int i = 0; i < np && count < max_placements; i++) {
        Placement* pl = &placements[i];
        SearchState s;
        memset(&s, 0, sizeof(s));
        memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
        lock_piece_on_board(s.board, board_height, active_piece, pl->rot, pl->landing_row, pl->col);
        int clears = clear_lines(s.board, board_height);
        bool pc = true;
        for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
        AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
        s.b2b = ar.new_b2b; s.combo = ar.new_combo;
        s.total_attack = ar.attack; s.max_single_attack = ar.attack;
        s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
        s.max_b2b_attack = s.b2b_attack;
        s.total_lines_cleared = clears; s.hold_piece = hold_piece;
        s.streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
        s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
        s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
        evaluate_state_decompose(&s, board_height, queue, queue_len,
                                  &decompose_out[count * NUM_DECOMPOSE]);
        count++;
    }

    // Hold piece placements
    if (hold_piece != 0) {
        np = find_placements(board_rows, board_height, hold_piece, placements, MAX_PLACEMENTS, NULL);
        for (int i = 0; i < np && count < max_placements; i++) {
            Placement* pl = &placements[i];
            SearchState s;
            memset(&s, 0, sizeof(s));
            memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s.board, board_height, hold_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s.board, board_height);
            bool pc = true;
            for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
            s.b2b = ar.new_b2b; s.combo = ar.new_combo;
            s.total_attack = ar.attack; s.max_single_attack = ar.attack;
            s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s.max_b2b_attack = s.b2b_attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
            s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
            evaluate_state_decompose(&s, board_height, queue, queue_len,
                                      &decompose_out[count * NUM_DECOMPOSE]);
            count++;
        }
    } else if (queue_len > 0) {
        int swap = queue[0];
        np = find_placements(board_rows, board_height, swap, placements, MAX_PLACEMENTS, NULL);
        for (int i = 0; i < np && count < max_placements; i++) {
            Placement* pl = &placements[i];
            SearchState s;
            memset(&s, 0, sizeof(s));
            memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s.board, board_height, swap, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s.board, board_height);
            bool pc = true;
            for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
            s.b2b = ar.new_b2b; s.combo = ar.new_combo;
            s.total_attack = ar.attack; s.max_single_attack = ar.attack;
            s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s.max_b2b_attack = s.b2b_attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
            s.next_queue_idx = 1; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
            evaluate_state_decompose(&s, board_height, queue, queue_len,
                                      &decompose_out[count * NUM_DECOMPOSE]);
            count++;
        }
    }

    return count;
}

// ============================================================
// Key Sequence Reconstruction (for depth-0 move output)
// ============================================================

static void b2b_write_sequence(const BFSStateMeta* meta, int bfs_state,
                               int is_hold, int max_len, int64_t* out_row) {
    int len = 0;
    int path[BFS_STATE_SPACE];
    int curr = bfs_state;

    while (meta[curr].parent != -1) {
        path[len++] = meta[curr].last_move;
        curr = meta[curr].parent;
    }

    int p = 0;
    out_row[p++] = KEY_START;
    if (is_hold) out_row[p++] = KEY_HOLD;

    // Reserve 1 slot for HARD_DROP — truncate path if needed
    int max_path_keys = max_len - p - 1;
    int start = (len > max_path_keys) ? (len - max_path_keys) : 0;
    for (int i = len - 1; i >= start; i--) {
        out_row[p++] = path[i];
    }

    out_row[p++] = KEY_HARD_DROP; // Always fits now
    while (p < max_len) out_row[p++] = KEY_PAD;
}

// ============================================================
// Beam Search Entry Point
// ============================================================

// Compare function for sorting beam by score (descending)
static int compare_states_desc(const void* a, const void* b) {
    float sa = ((const SearchState*)a)->score;
    float sb = ((const SearchState*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}

void b2b_search_c(
    const uint16_t* board_rows,     // Board bitmasks (board_height rows)
    int board_height,               // Typically 24
    int active_piece,               // Current active piece type (1-7)
    int hold_piece,                 // Current hold piece type (0=N/empty, 1-7)
    const int* queue,               // Piece types in queue
    int queue_len,                  // Number of pieces in queue
    int b2b,                        // Current b2b counter (-1 = none)
    int combo,                      // Current combo counter (-1 = none)
    int total_garbage,              // Total garbage lines in queue
    int garbage_push_delay,         // Ticks until garbage pushes (0 = immediate)
    int bag_seen_init,              // Bitmask: pieces consumed from current bag (after queue)
    int search_depth,               // Max search depth
    int beam_width,                 // Beam width
    int max_len,                    // Max key sequence length
    int* out_action_index,          // Output: action index
    int64_t* out_best_sequence      // Output: key sequence (length max_len)
) {
    if (!b2b_initialized) b2b_init_pieces();

    // Clamp parameters
    if (search_depth > MAX_SEARCH_DEPTH) search_depth = MAX_SEARCH_DEPTH;
    if (beam_width > MAX_BEAM_WIDTH) beam_width = MAX_BEAM_WIDTH;
    if (search_depth < 1) search_depth = 1;

    uint8_t initial_bag_seen = (uint8_t)(bag_seen_init & 0xFF);

    // Allocate beam arrays (current and next)
    // Use heap allocation since these can be large
    // Extra capacity when speculative depths are active (multiple pieces per beam state)
    int spec_mult = (search_depth > queue_len + 1) ? 2 : 1;
    int max_next = beam_width * MAX_PLACEMENTS * spec_mult;
    if (max_next > MAX_BEAM_WIDTH * MAX_PLACEMENTS * 2) max_next = MAX_BEAM_WIDTH * MAX_PLACEMENTS * 2;
    SearchState* curr_beam = (SearchState*)malloc(max_next * sizeof(SearchState));
    SearchState* next_beam = (SearchState*)malloc(max_next * sizeof(SearchState));
    int curr_beam_size = 0;
    int next_beam_size = 0;

    if (!curr_beam || !next_beam) {
        // Allocation failed
        *out_action_index = -1;
        for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        free(curr_beam);
        free(next_beam);
        return;
    }

    // Store depth-0 BFS meta and placement info for sequence reconstruction
    static BFSStateMeta depth0_meta_active[BFS_STATE_SPACE];
    static BFSStateMeta depth0_meta_hold[BFS_STATE_SPACE];
    static Placement depth0_placements[MAX_PLACEMENTS * 2];
    static int depth0_is_hold[MAX_PLACEMENTS * 2];
    int depth0_count = 0;

    // Initial garbage timer: the caller tells us the push delay.  A value
    // of 0 means garbage pushes immediately on the first non-clearing step;
    // 1 means it waits one tick, etc.  We use the delay as the initial
    // countdown for the simulated garbage timer in the search state.
    int init_garbage_timer = (garbage_push_delay > 0) ? garbage_push_delay : 0;

    // ---- Depth 0: enumerate placements for active piece and hold piece ----

    Placement placements[MAX_PLACEMENTS];
    int np;

    // Active piece placements
    np = find_placements(board_rows, board_height, active_piece, placements, MAX_PLACEMENTS,
                         depth0_meta_active);

    for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
        Placement* pl = &placements[i];
        SearchState* s = &next_beam[next_beam_size];

        memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
        lock_piece_on_board(s->board, board_height, active_piece, pl->rot, pl->landing_row, pl->col);
        int clears = clear_lines(s->board, board_height);

        bool perfect_clear = true;
        for (int r = 0; r < board_height; r++) {
            if (s->board[r] != 0) { perfect_clear = false; break; }
        }

        AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

        s->b2b = ar.new_b2b;
        s->combo = ar.new_combo;
        s->total_attack = ar.attack;
        s->max_single_attack = ar.attack;
        s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
        s->max_b2b_attack = s->b2b_attack;
        s->total_lines_cleared = clears;
        s->hold_piece = hold_piece;
        s->next_queue_idx = 0;
        s->depth0_placement_idx = depth0_count;
        s->b2b_broken = ar.b2b_broken;
        s->prev_b2b = b2b;
        s->streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
        s->bag_seen = initial_bag_seen;

        // Garbage simulation: cancel with attack, then tick+push if no clears
        {
            int gr = total_garbage;
            int gt = init_garbage_timer;
            if (ar.attack > 0 && gr > 0) {
                int cancel = (int)ar.attack;
                gr = (gr > cancel) ? gr - cancel : 0;
            }
            if (clears == 0 && gr > 0) {
                if (gt <= 0) {
                    push_simulated_garbage(s->board, board_height, gr);
                    gr = 0;
                } else {
                    gt--;
                }
            }
            s->garbage_remaining = gr;
            s->garbage_timer = gt;
        }

        s->score = evaluate_state(s, board_height, queue, queue_len);

        depth0_placements[depth0_count] = *pl;
        depth0_is_hold[depth0_count] = 0;
        depth0_count++;
        next_beam_size++;
    }

    // Hold piece placements
    if (hold_piece != PIECE_N) {
        // Swap: play hold piece, hold becomes active
        np = find_placements(board_rows, board_height, hold_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);

        for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
            Placement* pl = &placements[i];
            SearchState* s = &next_beam[next_beam_size];

            memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s->board, board_height, hold_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s->board, board_height);

            bool perfect_clear = true;
            for (int r = 0; r < board_height; r++) {
                if (s->board[r] != 0) { perfect_clear = false; break; }
            }

            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

            s->b2b = ar.new_b2b;
            s->combo = ar.new_combo;
            s->total_attack = ar.attack;
            s->max_single_attack = ar.attack;
            s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s->max_b2b_attack = s->b2b_attack;
            s->total_lines_cleared = clears;
            s->hold_piece = active_piece;
            s->next_queue_idx = 0;
            s->depth0_placement_idx = depth0_count;
            s->b2b_broken = ar.b2b_broken;
            s->prev_b2b = b2b;
            s->streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
            s->bag_seen = initial_bag_seen;

            // Garbage simulation
            {
                int gr = total_garbage;
                int gt = init_garbage_timer;
                if (ar.attack > 0 && gr > 0) {
                    int cancel = (int)ar.attack;
                    gr = (gr > cancel) ? gr - cancel : 0;
                }
                if (clears == 0 && gr > 0) {
                    if (gt <= 0) {
                        push_simulated_garbage(s->board, board_height, gr);
                        gr = 0;
                    } else {
                        gt--;
                    }
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
            }

            s->score = evaluate_state(s, board_height, queue, queue_len);

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            next_beam_size++;
        }
    } else if (queue_len > 0) {
        // No hold piece yet — hold swaps active with first queue piece
        int swap_piece = queue[0];
        np = find_placements(board_rows, board_height, swap_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);

        for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
            Placement* pl = &placements[i];
            SearchState* s = &next_beam[next_beam_size];

            memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s->board, board_height, swap_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s->board, board_height);

            bool perfect_clear = true;
            for (int r = 0; r < board_height; r++) {
                if (s->board[r] != 0) { perfect_clear = false; break; }
            }

            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

            s->b2b = ar.new_b2b;
            s->combo = ar.new_combo;
            s->total_attack = ar.attack;
            s->max_single_attack = ar.attack;
            s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s->max_b2b_attack = s->b2b_attack;
            s->total_lines_cleared = clears;
            s->hold_piece = active_piece;
            s->next_queue_idx = 1;
            s->depth0_placement_idx = depth0_count;
            s->b2b_broken = ar.b2b_broken;
            s->prev_b2b = b2b;
            s->streak_attack = (ar.attack > 0) ? ar.attack : 0.0f;
            s->bag_seen = initial_bag_seen;

            // Garbage simulation
            {
                int gr = total_garbage;
                int gt = init_garbage_timer;
                if (ar.attack > 0 && gr > 0) {
                    int cancel = (int)ar.attack;
                    gr = (gr > cancel) ? gr - cancel : 0;
                }
                if (clears == 0 && gr > 0) {
                    if (gt <= 0) {
                        push_simulated_garbage(s->board, board_height, gr);
                        gr = 0;
                    } else {
                        gt--;
                    }
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
            }

            s->score = evaluate_state(s, board_height, queue, queue_len);

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            next_beam_size++;
        }
    }

    // Sort and trim to beam_width
    if (next_beam_size > beam_width) {
        qsort(next_beam, next_beam_size, sizeof(SearchState), compare_states_desc);
        next_beam_size = beam_width;
    }

    // Swap beams
    {
        SearchState* tmp = curr_beam;
        curr_beam = next_beam;
        curr_beam_size = next_beam_size;
        next_beam = tmp;
        next_beam_size = 0;
    }

    // ---- Depths 1..search_depth-1 ----
    for (int depth = 1; depth < search_depth; depth++) {
        next_beam_size = 0;

        for (int bi = 0; bi < curr_beam_size; bi++) {
            SearchState* parent = &curr_beam[bi];

            int qi = parent->next_queue_idx;
            if (qi >= queue_len) {
                // ---- Speculative: branch on remaining bag pieces ----
                int remaining[7];
                int n_remaining = bag_get_remaining(parent->bag_seen, remaining);

                if (n_remaining == 0) {
                    // All pieces consumed — shouldn't happen, keep as leaf
                    if (next_beam_size < max_next) {
                        next_beam[next_beam_size] = *parent;
                        next_beam_size++;
                    }
                    continue;
                }

                for (int pi = 0; pi < n_remaining; pi++) {
                    int spec_piece = remaining[pi];
                    uint8_t new_bag = bag_consume_piece(parent->bag_seen, spec_piece);

                    // Try placing speculative piece directly
                    np = find_placements(parent->board, board_height, spec_piece, placements, MAX_PLACEMENTS, NULL);

                    for (int i = 0; i < np && next_beam_size < max_next; i++) {
                        Placement* pl = &placements[i];
                        SearchState* s = &next_beam[next_beam_size];

                        memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                        lock_piece_on_board(s->board, board_height, spec_piece, pl->rot, pl->landing_row, pl->col);
                        int clears = clear_lines(s->board, board_height);

                        bool perfect_clear = true;
                        for (int r = 0; r < board_height; r++) {
                            if (s->board[r] != 0) { perfect_clear = false; break; }
                        }

                        AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                        s->b2b = ar.new_b2b;
                        s->combo = ar.new_combo;
                        s->total_attack = parent->total_attack + ar.attack;
                        s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                        { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                          s->b2b_attack = parent->b2b_attack + ba;
                          s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                        s->total_lines_cleared = parent->total_lines_cleared + clears;
                        s->hold_piece = parent->hold_piece;
                        s->next_queue_idx = qi + 1;
                        s->depth0_placement_idx = parent->depth0_placement_idx;
                        s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                        s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                        s->streak_attack = (ar.attack > 0) ? parent->streak_attack + ar.attack : 0.0f;
                        s->bag_seen = new_bag;

                        // Garbage simulation
                        {
                            int gr = parent->garbage_remaining;
                            int gt = parent->garbage_timer;
                            if (ar.attack > 0 && gr > 0) {
                                int cancel = (int)ar.attack;
                                gr = (gr > cancel) ? gr - cancel : 0;
                            }
                            if (clears == 0 && gr > 0) {
                                if (gt <= 0) {
                                    push_simulated_garbage(s->board, board_height, gr);
                                    gr = 0;
                                } else {
                                    gt--;
                                }
                            }
                            s->garbage_remaining = gr;
                            s->garbage_timer = gt;
                        }

                        s->score = evaluate_state(s, board_height, queue, queue_len);

                        next_beam_size++;
                    }

                    // Try hold swap (if hold piece exists)
                    if (parent->hold_piece != PIECE_N) {
                        int held = parent->hold_piece;
                        np = find_placements(parent->board, board_height, held, placements, MAX_PLACEMENTS, NULL);

                        for (int i = 0; i < np && next_beam_size < max_next; i++) {
                            Placement* pl = &placements[i];
                            SearchState* s = &next_beam[next_beam_size];

                            memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                            lock_piece_on_board(s->board, board_height, held, pl->rot, pl->landing_row, pl->col);
                            int clears = clear_lines(s->board, board_height);

                            bool perfect_clear = true;
                            for (int r = 0; r < board_height; r++) {
                                if (s->board[r] != 0) { perfect_clear = false; break; }
                            }

                            AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                            s->b2b = ar.new_b2b;
                            s->combo = ar.new_combo;
                            s->total_attack = parent->total_attack + ar.attack;
                            s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                            { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                              s->b2b_attack = parent->b2b_attack + ba;
                              s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                            s->total_lines_cleared = parent->total_lines_cleared + clears;
                            s->hold_piece = spec_piece; // speculative piece goes to hold
                            s->next_queue_idx = qi + 1;
                            s->depth0_placement_idx = parent->depth0_placement_idx;
                            s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                            s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                            s->streak_attack = (ar.attack > 0) ? parent->streak_attack + ar.attack : 0.0f;
                            s->bag_seen = new_bag;

                            // Garbage simulation
                            {
                                int gr = parent->garbage_remaining;
                                int gt = parent->garbage_timer;
                                if (ar.attack > 0 && gr > 0) {
                                    int cancel = (int)ar.attack;
                                    gr = (gr > cancel) ? gr - cancel : 0;
                                }
                                if (clears == 0 && gr > 0) {
                                    if (gt <= 0) {
                                        push_simulated_garbage(s->board, board_height, gr);
                                        gr = 0;
                                    } else {
                                        gt--;
                                    }
                                }
                                s->garbage_remaining = gr;
                                s->garbage_timer = gt;
                            }

                            s->score = evaluate_state(s, board_height, queue, queue_len);

                            next_beam_size++;
                        }
                    }
                }
                continue;
            }

            int piece = queue[qi];

            // Try placing active piece (no hold)
            np = find_placements(parent->board, board_height, piece, placements, MAX_PLACEMENTS, NULL);

            for (int i = 0; i < np && next_beam_size < max_next; i++) {
                Placement* pl = &placements[i];
                SearchState* s = &next_beam[next_beam_size];

                memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                lock_piece_on_board(s->board, board_height, piece, pl->rot, pl->landing_row, pl->col);
                int clears = clear_lines(s->board, board_height);

                bool perfect_clear = true;
                for (int r = 0; r < board_height; r++) {
                    if (s->board[r] != 0) { perfect_clear = false; break; }
                }

                AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                s->b2b = ar.new_b2b;
                s->combo = ar.new_combo;
                s->total_attack = parent->total_attack + ar.attack;
                s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                  s->b2b_attack = parent->b2b_attack + ba;
                  s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                s->total_lines_cleared = parent->total_lines_cleared + clears;
                s->hold_piece = parent->hold_piece;
                s->next_queue_idx = qi + 1;
                s->depth0_placement_idx = parent->depth0_placement_idx;
                s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                s->streak_attack = (ar.attack > 0) ? parent->streak_attack + ar.attack : 0.0f;
                s->bag_seen = parent->bag_seen;

                // Garbage simulation (inherited from parent)
                {
                    int gr = parent->garbage_remaining;
                    int gt = parent->garbage_timer;
                    if (ar.attack > 0 && gr > 0) {
                        int cancel = (int)ar.attack;
                        gr = (gr > cancel) ? gr - cancel : 0;
                    }
                    if (clears == 0 && gr > 0) {
                        if (gt <= 0) {
                            push_simulated_garbage(s->board, board_height, gr);
                            gr = 0;
                        } else {
                            gt--;
                        }
                    }
                    s->garbage_remaining = gr;
                    s->garbage_timer = gt;
                }

                s->score = evaluate_state(s, board_height, queue, queue_len);

                next_beam_size++;
            }

            // Try hold swap
            if (parent->hold_piece != PIECE_N) {
                int held = parent->hold_piece;
                np = find_placements(parent->board, board_height, held, placements, MAX_PLACEMENTS, NULL);

                for (int i = 0; i < np && next_beam_size < max_next; i++) {
                    Placement* pl = &placements[i];
                    SearchState* s = &next_beam[next_beam_size];

                    memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                    lock_piece_on_board(s->board, board_height, held, pl->rot, pl->landing_row, pl->col);
                    int clears = clear_lines(s->board, board_height);

                    bool perfect_clear = true;
                    for (int r = 0; r < board_height; r++) {
                        if (s->board[r] != 0) { perfect_clear = false; break; }
                    }

                    AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                    s->b2b = ar.new_b2b;
                    s->combo = ar.new_combo;
                    s->total_attack = parent->total_attack + ar.attack;
                    s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                    { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                      s->b2b_attack = parent->b2b_attack + ba;
                      s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                    s->total_lines_cleared = parent->total_lines_cleared + clears;
                    s->hold_piece = piece;
                    s->next_queue_idx = qi + 1;
                    s->depth0_placement_idx = parent->depth0_placement_idx;
                    s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                    s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                    s->streak_attack = (ar.attack > 0) ? parent->streak_attack + ar.attack : 0.0f;
                    s->bag_seen = parent->bag_seen;

                    // Garbage simulation (inherited from parent)
                    {
                        int gr = parent->garbage_remaining;
                        int gt = parent->garbage_timer;
                        if (ar.attack > 0 && gr > 0) {
                            int cancel = (int)ar.attack;
                            gr = (gr > cancel) ? gr - cancel : 0;
                        }
                        if (clears == 0 && gr > 0) {
                            if (gt <= 0) {
                                push_simulated_garbage(s->board, board_height, gr);
                                gr = 0;
                            } else {
                                gt--;
                            }
                        }
                        s->garbage_remaining = gr;
                        s->garbage_timer = gt;
                    }

                    s->score = evaluate_state(s, board_height, queue, queue_len);

                    next_beam_size++;
                }
            } else if (qi + 1 < queue_len) {
                // No hold piece — holding puts active into hold, play next from queue
                int next_piece = piece; // Goes to hold
                int play_piece_idx = qi + 1;
                int play_piece = queue[play_piece_idx];

                np = find_placements(parent->board, board_height, play_piece, placements, MAX_PLACEMENTS, NULL);

                for (int i = 0; i < np && next_beam_size < max_next; i++) {
                    Placement* pl = &placements[i];
                    SearchState* s = &next_beam[next_beam_size];

                    memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                    lock_piece_on_board(s->board, board_height, play_piece, pl->rot, pl->landing_row, pl->col);
                    int clears = clear_lines(s->board, board_height);

                    bool perfect_clear = true;
                    for (int r = 0; r < board_height; r++) {
                        if (s->board[r] != 0) { perfect_clear = false; break; }
                    }

                    AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                    s->b2b = ar.new_b2b;
                    s->combo = ar.new_combo;
                    s->total_attack = parent->total_attack + ar.attack;
                    s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                    { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                      s->b2b_attack = parent->b2b_attack + ba;
                      s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                    s->total_lines_cleared = parent->total_lines_cleared + clears;
                    s->hold_piece = next_piece;
                    s->next_queue_idx = play_piece_idx + 1;
                    s->depth0_placement_idx = parent->depth0_placement_idx;
                    s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                    s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                    s->streak_attack = (ar.attack > 0) ? parent->streak_attack + ar.attack : 0.0f;
                    s->bag_seen = parent->bag_seen;

                    // Garbage simulation (inherited from parent)
                    {
                        int gr = parent->garbage_remaining;
                        int gt = parent->garbage_timer;
                        if (ar.attack > 0 && gr > 0) {
                            int cancel = (int)ar.attack;
                            gr = (gr > cancel) ? gr - cancel : 0;
                        }
                        if (clears == 0 && gr > 0) {
                            if (gt <= 0) {
                                push_simulated_garbage(s->board, board_height, gr);
                                gr = 0;
                            } else {
                                gt--;
                            }
                        }
                        s->garbage_remaining = gr;
                        s->garbage_timer = gt;
                    }

                    s->score = evaluate_state(s, board_height, queue, queue_len);

                    next_beam_size++;
                }
            }
        }

        // Sort and trim
        if (next_beam_size > beam_width) {
            qsort(next_beam, next_beam_size, sizeof(SearchState), compare_states_desc);
            next_beam_size = beam_width;
        }

        // Swap beams
        {
            SearchState* tmp = curr_beam;
            curr_beam = next_beam;
            curr_beam_size = next_beam_size;
            next_beam = tmp;
            next_beam_size = 0;
        }
    }

    // ---- Extract best result ----
    if (curr_beam_size == 0) {
        *out_action_index = -1;
        for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        free(curr_beam);
        free(next_beam);
        return;
    }

    // Find best state
    int best_idx = 0;
    for (int i = 1; i < curr_beam_size; i++) {
        if (curr_beam[i].score > curr_beam[best_idx].score) best_idx = i;
    }

    int d0_idx = curr_beam[best_idx].depth0_placement_idx;
    Placement* best_pl = &depth0_placements[d0_idx];
    int is_hold = depth0_is_hold[d0_idx];

    // Compute action index: hold * 160 + rot * 40 + norm_col * 4 + spin_type
    int played_piece;
    if (!is_hold) {
        played_piece = active_piece;
    } else {
        played_piece = (hold_piece != PIECE_N) ? hold_piece : queue[0];
    }
    int norm_col = best_pl->col + B2B_PIECES[played_piece].orientations[best_pl->rot].min_col;
    *out_action_index = is_hold * 160 + best_pl->rot * 40 + norm_col * 4 + best_pl->spin_type;

    // Store full placement info for C game loop (avoids lossy action_idx round-trip)
    b2b_last_placement.rot = best_pl->rot;
    b2b_last_placement.col = best_pl->col;
    b2b_last_placement.landing_row = best_pl->landing_row;
    b2b_last_placement.spin_type = best_pl->spin_type;

    // Reconstruct key sequence
    BFSStateMeta* meta_src = is_hold ? depth0_meta_hold : depth0_meta_active;
    b2b_write_sequence(meta_src, best_pl->bfs_state, is_hold, max_len, out_best_sequence);

    free(curr_beam);
    free(next_beam);
}

// ============================================================
// Full Game Loop in C (for optimizer — no Python overhead)
// ============================================================

// --- TetrioRNG: exact replica of TetrioRandom.py ---

typedef struct {
    int64_t t;
} TetrioRNG;

static void rng_init(TetrioRNG* rng, int seed) {
    int64_t t = seed % 2147483647;
    if (t <= 0) t += 2147483646;
    rng->t = t;
}

static int rng_next_int(TetrioRNG* rng) {
    rng->t = (16807 * rng->t) % 2147483647;
    return (int)rng->t;
}

static float rng_next_float(TetrioRNG* rng) {
    return (float)(rng_next_int(rng) - 1) / 2147483646.0f;
}

// Tetromino order matching Python: [Z, L, O, S, I, J, T]
static void rng_next_bag(TetrioRNG* rng, int* bag) {
    int base[7] = {PIECE_Z, PIECE_L, PIECE_O, PIECE_S,
                   PIECE_I, PIECE_J, PIECE_T};
    memcpy(bag, base, sizeof(base));
    // Fisher-Yates shuffle (matching Python's while i > 0 loop)
    for (int i = 6; i > 0; i--) {
        int j = (int)(rng_next_float(rng) * (i + 1));
        int tmp = bag[i]; bag[i] = bag[j]; bag[j] = tmp;
    }
}

// --- Simple xorshift RNG for garbage generation ---

typedef struct {
    uint32_t state;
} SimpleRNG;

static void srng_init(SimpleRNG* rng, uint32_t seed) {
    rng->state = seed ? seed : 1;
}

static uint32_t srng_next(SimpleRNG* rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static float srng_float(SimpleRNG* rng) {
    return (float)(srng_next(rng) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// --- Piece queue (bag-based) ---

#define BAG_SIZE 7
#define MAX_QUEUE 16

typedef struct {
    TetrioRNG rng;
    int bag[BAG_SIZE];
    int bag_pos;
    int queue[MAX_QUEUE];
    int queue_len;
    int queue_size;
} PieceQueue;

static void pq_init(PieceQueue* pq, int seed, int qsize) {
    rng_init(&pq->rng, seed);
    pq->bag_pos = BAG_SIZE; // force first bag generation
    pq->queue_len = 0;
    pq->queue_size = qsize;
}

static int pq_next_piece(PieceQueue* pq) {
    if (pq->bag_pos >= BAG_SIZE) {
        rng_next_bag(&pq->rng, pq->bag);
        pq->bag_pos = 0;
    }
    return pq->bag[pq->bag_pos++];
}

static void pq_fill(PieceQueue* pq) {
    while (pq->queue_len < pq->queue_size) {
        pq->queue[pq->queue_len++] = pq_next_piece(pq);
    }
}

static int pq_pop(PieceQueue* pq) {
    if (pq->queue_len <= 0) return PIECE_N;
    int piece = pq->queue[0];
    for (int i = 0; i < pq->queue_len - 1; i++)
        pq->queue[i] = pq->queue[i + 1];
    pq->queue_len--;
    return piece;
}

// --- Garbage queue ---

#define MAX_GARB_ENTRIES 32

typedef struct {
    int rows;
    int col;
    int timer;
} GarbEntry;

static void garb_cancel(GarbEntry* gq, int* cnt, int attack) {
    int rem = attack;
    while (rem > 0 && *cnt > 0) {
        if (gq[0].rows <= rem) {
            rem -= gq[0].rows;
            for (int i = 0; i < *cnt - 1; i++) gq[i] = gq[i + 1];
            (*cnt)--;
        } else {
            gq[0].rows -= rem;
            rem = 0;
        }
    }
}

static void garb_tick(GarbEntry* gq, int cnt) {
    for (int i = 0; i < cnt; i++) {
        if (gq[i].timer > 0) gq[i].timer--;
    }
}

static bool garb_push_one(uint16_t* board, int bh, GarbEntry* gq, int* cnt) {
    if (*cnt <= 0 || gq[0].timer > 0) return false;
    uint16_t full = (1 << BOARD_COLS) - 1;
    int rows = gq[0].rows;
    int col = gq[0].col;
    for (int r = 0; r < bh - rows; r++) board[r] = board[r + rows];
    uint16_t garb_row = full & ~(1 << col);
    for (int r = bh - rows; r < bh; r++) board[r] = garb_row;
    for (int i = 0; i < *cnt - 1; i++) gq[i] = gq[i + 1];
    (*cnt)--;
    return true;
}

static void garb_push_all(uint16_t* board, int bh, GarbEntry* gq, int* cnt) {
    while (garb_push_one(board, bh, gq, cnt)) {}
}

static int garb_total(const GarbEntry* gq, int cnt) {
    int t = 0;
    for (int i = 0; i < cnt; i++) t += gq[i].rows;
    return t;
}

// --- Game config & result structs (exported) ---

typedef struct {
    int seed;
    float garbage_chance;
    int garbage_min;
    int garbage_max;
    int garbage_push_delay;
} GameConfig;

typedef struct {
    int steps_completed;
    int survived;
    float total_attack;
    int max_b2b;
    int end_height;
    float avg_height;
    int max_height;
} GameResult;

// --- Game loop entry point ---

void b2b_run_eval_games(
    int num_games,
    const GameConfig* configs,
    int num_steps,
    int search_depth,
    int beam_width,
    int queue_size,
    GameResult* results
) {
    if (!b2b_initialized) b2b_init_pieces();

    for (int g = 0; g < num_games; g++) {
        GameConfig cfg = configs[g];
        GameResult* res = &results[g];

        // --- Init game state ---
        uint16_t board[BOARD_ROWS];
        memset(board, 0, sizeof(board));
        int bh = 24; // standard board height

        PieceQueue pq;
        pq_init(&pq, cfg.seed, queue_size);
        pq_fill(&pq);
        int active_piece = pq_pop(&pq);
        pq_fill(&pq);
        int hold_piece = PIECE_N;
        int b2b = -1, combo = -1;

        SimpleRNG grng;
        srng_init(&grng, (uint32_t)(cfg.seed * 7 + 12345));
        GarbEntry gq[MAX_GARB_ENTRIES];
        int gcnt = 0;

        float total_attack = 0.0f;
        int max_b2b_val = 0;
        float height_sum = 0.0f;
        int last_height = 0, peak_height = 0;
        int steps_done = 0;
        bool died = false;

        for (int step = 0; step < num_steps; step++) {
            if (b2b > max_b2b_val) max_b2b_val = b2b;

            int tg = garb_total(gq, gcnt);

            // --- Compute bag_seen for speculative search ---
            // pq.bag[0..bag_pos-1] are pieces already drawn from current bag
            uint8_t cur_bag_seen = 0;
            if (pq.bag_pos < BAG_SIZE) {
                for (int bi2 = 0; bi2 < pq.bag_pos; bi2++) {
                    cur_bag_seen |= (uint8_t)(1 << pq.bag[bi2]);
                }
            }
            // If bag_pos >= BAG_SIZE, cur_bag_seen stays 0 (fresh bag)

            // --- Beam search (reuses b2b_search_c — sequence is discarded) ---
            int action_idx;
            int64_t dummy_seq[15];
            b2b_search_c(
                board, bh, active_piece, hold_piece,
                pq.queue, pq.queue_len, b2b, combo, tg,
                cfg.garbage_push_delay,
                (int)cur_bag_seen,
                search_depth, beam_width, 15,
                &action_idx, dummy_seq
            );

            if (action_idx < 0) { died = true; break; }

            // --- Read placement directly from b2b_search_c's result ---
            // (avoids lossy action_idx round-trip — preserves BFS landing row)
            int is_hold   = action_idx / 160;
            int rot       = b2b_last_placement.rot;
            int col       = b2b_last_placement.col;
            int lr        = b2b_last_placement.landing_row;
            int spin_type = b2b_last_placement.spin_type;

            // Determine played piece & update hold/queue
            int played;
            if (!is_hold) {
                played = active_piece;
                active_piece = pq_pop(&pq);
            } else if (hold_piece != PIECE_N) {
                played = hold_piece;
                hold_piece = active_piece;
                active_piece = pq_pop(&pq);
            } else {
                hold_piece = active_piece;
                played = pq_pop(&pq);
                active_piece = pq_pop(&pq);
            }
            pq_fill(&pq);

            // --- Apply placement ---
            lock_piece_on_board(board, bh, played, rot, lr, col);
            int clears = clear_lines(board, bh);

            bool pc = true;
            for (int r = 0; r < bh; r++) {
                if (board[r] != 0) { pc = false; break; }
            }

            AttackResult ar = compute_attack(clears, spin_type, b2b, combo, pc);
            total_attack += ar.attack;
            b2b = ar.new_b2b;
            combo = ar.new_combo;

            // --- Garbage handling ---
            if (ar.attack > 0) garb_cancel(gq, &gcnt, (int)ar.attack);

            if (clears == 0) {
                garb_tick(gq, gcnt);
                garb_push_one(board, bh, gq, &gcnt); // one tier per step
            }

            // Generate new garbage
            if (cfg.garbage_chance > 0.0f && cfg.garbage_max > 0) {
                if (srng_float(&grng) <= cfg.garbage_chance) {
                    int nr;
                    if (cfg.garbage_min == cfg.garbage_max) {
                        nr = cfg.garbage_min;
                    } else {
                        nr = cfg.garbage_min +
                             (int)(srng_float(&grng) *
                                   (cfg.garbage_max - cfg.garbage_min + 1));
                        if (nr > cfg.garbage_max) nr = cfg.garbage_max;
                    }
                    if (nr > 0 && gcnt < MAX_GARB_ENTRIES) {
                        int gap = (int)(srng_float(&grng) * BOARD_COLS);
                        if (gap >= BOARD_COLS) gap = BOARD_COLS - 1;
                        gq[gcnt].rows = nr;
                        gq[gcnt].col = gap;
                        gq[gcnt].timer = cfg.garbage_push_delay;
                        gcnt++;
                    }
                }
            }

            // Immediate push for delay=0
            if (cfg.garbage_push_delay == 0) {
                garb_push_all(board, bh, gq, &gcnt);
            }

            // --- Track stats ---
            int h = 0;
            for (int r = 0; r < bh; r++) {
                if (board[r] != 0) { h = bh - r; break; }
            }
            height_sum += (float)h;
            last_height = h;
            if (h > peak_height) peak_height = h;
            steps_done++;

            // --- Death check (rows 0..3 for bh=24) ---
            for (int r = 0; r < bh - 20; r++) {
                if (board[r] != 0) { died = true; break; }
            }
            if (died) break;
        }

        // Check final b2b
        if (b2b > max_b2b_val) max_b2b_val = b2b;

        res->steps_completed = steps_done;
        res->survived = died ? 0 : 1;
        res->total_attack = total_attack;
        res->max_b2b = max_b2b_val;
        res->end_height = died ? 20 : last_height;
        res->avg_height = (steps_done > 0) ? height_sum / (float)steps_done : 0.0f;
        res->max_height = peak_height;
    }
}
