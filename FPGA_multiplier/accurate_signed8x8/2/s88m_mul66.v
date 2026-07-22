// Exact unsigned 6x6 multiplier, following unsigned8x8/8x8_50.
// Resources: 27 LUTs (11 LUT6_2 + 16 LUT6) + 2 CARRY4.
module s88m_mul66 (
    input  wire [5:0] a,
    input  wire [5:0] b,
    input  wire       shared_sign,
    output wire [11:0] prod,
    output wire       shared_high_n
);

wire [5:0] hh;
wire [5:0] hl;
wire [5:0] lh;
wire [5:0] ll;

s88m_mul33 mul_hh (.a(a[5:3]), .b(b[5:3]), .prod(hh));
s88m_mul33 mul_hl (.a(a[5:3]), .b(b[2:0]), .prod(hl));
s88m_mul33 mul_lh (.a(a[2:0]), .b(b[5:3]), .prod(lh));
s88m_mul33 mul_ll (.a(a[2:0]), .b(b[2:0]), .prod(ll));

s88m_comp66 compressor (
    .hh(hh), .hl(hl), .lh(lh), .ll(ll),
    .shared_x5(b[5]), .shared_sign(shared_sign),
    .prod(prod), .shared_high_n(shared_high_n)
);

endmodule
