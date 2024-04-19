#define MASKTop 11
#define SHIFTS (MASKTop >> 1)
#define TILE_WIDTH 16
#define BLOCK_SIZE (TILE_WIDTH + MASKTop - 1)
#define SHIFTScol (BLOCK_SIZE >> 1)