#ifndef TET_CORE_H
#define TET_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TetState TetState;

TetState* tet_state_new(int seed, int board_height, int queue_size,
                        int garbage_push_delay);
void      tet_state_free(TetState* s);
TetState* tet_state_clone(const TetState* s);

int       tet_state_serialized_size(void);
int       tet_state_serialize(const TetState* s, uint8_t* out);
TetState* tet_state_deserialize(const uint8_t* buf, int buf_len);

void tet_state_get_board(const TetState* s, uint16_t* out_board);
int  tet_state_get_board_height(const TetState* s);
int  tet_state_get_active_piece(const TetState* s);
int  tet_state_get_hold_piece(const TetState* s);
int  tet_state_get_queue(const TetState* s, int* out_queue, int max_len);
int  tet_state_get_b2b(const TetState* s);
int  tet_state_get_combo(const TetState* s);
int  tet_state_get_total_garbage(const TetState* s);

// Result of applying a placement.
typedef struct TetEvent {
    int   clears;          // lines cleared (0..4)
    float attack;          // outgoing attack
    int   new_b2b;         // b2b counter after the placement
    int   new_combo;       // combo counter after the placement
    int   spin_type;       // SPIN_NONE / T_MINI / T_FULL / ALL_MINI as supplied
    int   perfect_clear;   // 0/1
    int   terminal;        // 0/1: any cell in the death zone after garbage
    int   garbage_pushed;  // 0/1: did this step push at least one garbage row
} TetEvent;

// Enumerate all reachable placements for the current active piece, and
// optionally for the hold piece (or queue[0] if hold is empty).
//
// out_buf layout: row-major [n × 5] int32, columns
//   [is_hold, rot, col, landing_row, spin_type]
// max_placements is the row capacity; the function never writes more than
// max_placements rows.  Returns the number of rows written.  Returns 0 if
// the active piece cannot spawn (top-out reachable from caller).
int tet_enumerate_placements(const TetState* s, int32_t* out_buf,
                             int max_placements, int include_hold);

// Apply a placement.  Mutates state in-place: locks the played piece, clears
// lines, updates b2b/combo, cancels own garbage with the attack, ticks
// garbage timers and pushes one ready row when no lines were cleared, and
// (if push_delay==0) flushes the rest of the garbage queue.
//
// out_event may be NULL.  Returns 0 on success, nonzero on error.
//
// Precondition: the placement (is_hold, rot, col, landing_row, spin_type)
// was produced by tet_enumerate_placements on the same state — no validation
// is performed.
int tet_apply_placement(TetState* s,
                        int is_hold, int rot, int col,
                        int landing_row, int spin_type,
                        TetEvent* out_event);

// Read the (rot, col, landing_row, spin_type) of the last placement chosen
// by b2b_search_c.  Helper for cross-validation: action_idx alone does not
// fully determine the placement (the BFS landing row is not encoded).
void tet_get_last_search_placement(int* out_rot, int* out_col,
                                   int* out_landing_row, int* out_spin_type);

// Decompose the depth-0 placement scores into the 21 hand-tuned components.
// Wraps b2b_decompose_c using the state's internal fields.  out_buf must be
// at least max_placements * 21 floats.  Returns the number of placements
// scored (rows in out_buf).
int tet_decompose(const TetState* s, float* out_buf, int max_placements);

// Look up the bounding-box-to-board column offset for a (piece_type, rot)
// combination.  Used to normalize the placement col returned by
// tet_enumerate_placements into a 0..9 board column.  Returns 0 for invalid
// inputs.
int tet_get_piece_min_col(int piece_type, int rotation);

// Stochastically inject ambient garbage into the state's garbage queue.
// Uses the state's internal SimpleRNG (seeded deterministically at
// construction), so calls reproduce given the same seed.  Mirrors the
// generation logic in b2b_run_eval_games:
//   - With probability `chance`, add a single garbage entry whose row count
//     is uniform over [min_rows, max_rows] (inclusive) and whose hole column
//     is uniform over [0, BOARD_COLS).
//   - Timer is set to the state's garbage_push_delay so the new entry waits
//     to be pushed onto the board.
// Returns 1 if a new entry was added, 0 otherwise (rolled "no", queue full,
// or invalid args).
int tet_inject_random_garbage(TetState* s, float chance,
                              int min_rows, int max_rows);

#ifdef __cplusplus
}
#endif
#endif
