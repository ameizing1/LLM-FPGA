// Signed two's-complement 8x8 multiplier using only LUT6_2 and CARRY4
// in the combinational datapath. No DSP or inferred arithmetic operator.
//
// Expected structural resources before device-specific optimization:
//   47 LUT6_2 + 10 CARRY4
module signed88 (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);

wire [1:0] ah;
wire [1:0] bh;
wire [5:0] al;
wire [5:0] bl;

assign ah = a[7:6];
assign bh = b[7:6];
assign al = a[5:0];
assign bl = b[5:0];

wire [11:0] ll;
wire [7:0]  hl_bias;
wire [7:0]  lh_bias;

wire ll_high_1;
wire ll_high_2;
wire ll_high_3;
wire hl_high_n;
wire lh_high_n;
wire hh0;
wire hh1;
wire hh2_bw;

// Shared parent LUTs provide the otherwise separate final row bits of the
// 6x2 children and the encoded 2x2 signed high product.
LUT6_2 #(
    .INIT(64'ha0a0a0a088888888)
) shared_lut_1 (
    .I0(a[5]), .I1(b[1]), .I2(b[3]), .I3(1'b1),
    .I4(1'b1), .I5(1'b1),
    .O6(ll_high_2), .O5(ll_high_1)
);

LUT6_2 #(
    .INIT(64'h3f3f3f3f88888888)
) shared_lut_2 (
    .I0(a[5]), .I1(b[5]), .I2(a[7]), .I3(1'b1),
    .I4(1'b1), .I5(1'b1),
    .O6(hl_high_n), .O5(ll_high_3)
);

LUT6_2 #(
    .INIT(64'hf000f00077777777)
) shared_lut_3 (
    .I0(a[5]), .I1(b[7]), .I2(a[6]), .I3(b[6]),
    .I4(1'b1), .I5(1'b1),
    .O6(hh0), .O5(lh_high_n)
);

// x = a[7]&b[6], y = a[6]&b[7]
// O5 = x^y; O6 = (~x)&(~y), the BW carry encoding used by signed88_comp.
LUT6_2 #(
    .INIT(64'h0777077778887888)
) shared_lut_4 (
    .I0(a[7]), .I1(b[6]), .I2(a[6]), .I3(b[7]),
    .I4(1'b1), .I5(1'b1),
    .O6(hh2_bw), .O5(hh1)
);

signed88_u66 ll_multiplier (
    .a(al), .b(bl),
    .high_1(ll_high_1), .high_2(ll_high_2), .high_3(ll_high_3),
    .prod(ll)
);

// ah and bh are interpreted as signed two-bit values by signed88_bw62.
signed88_bw62 hl_multiplier (
    .x(bl), .y(ah), .high_n(hl_high_n), .biased_prod(hl_bias)
);

signed88_bw62 lh_multiplier (
    .x(al), .y(bh), .high_n(lh_high_n), .biased_prod(lh_bias)
);

wire [4:0] hh;
assign hh = {a[7], b[7], hh2_bw, hh1, hh0};

signed88_comp final_compressor (
    .hh(hh), .hl_bias(hl_bias), .lh_bias(lh_bias), .ll(ll), .prod(prod)
);

endmodule
