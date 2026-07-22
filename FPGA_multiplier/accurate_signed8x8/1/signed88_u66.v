// Unsigned 6x6 LL multiplier, split into three unsigned 6x2 blocks.
// Resources including signed88_comp66: 24 LUT6_2 + 5 CARRY4.
module signed88_u66 (
    input  wire [5:0]  a,
    input  wire [5:0]  b,
    input  wire        high_1,
    input  wire        high_2,
    input  wire        high_3,
    output wire [11:0] prod
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

signed88_u62 low_block (
    .a(a), .b(b[1:0]), .high(high_1), .prod(plow)
);

signed88_u62 mid_block (
    .a(a), .b(b[3:2]), .high(high_2), .prod(pmid)
);

signed88_u62 high_block (
    .a(a), .b(b[5:4]), .high(high_3), .prod(phigh)
);

signed88_comp66 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh), .prod(prod)
);

endmodule
