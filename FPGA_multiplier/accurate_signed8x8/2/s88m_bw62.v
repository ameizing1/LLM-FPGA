// Exact signed 6x2 Baugh-Wooley block.
//
// x is unsigned; y represents the signed value y[0] - 2*y[1].
// The unsigned result has a deliberate constant bias:
//
//     biased_prod = x * signed(y) + 128
//
// The parent creates high_n = ~(x[5] & y[1]); this lets two instances share
// one physical LUT. The two +128 biases are absorbed by the final compressor.
// Resources inside this module: 5 LUT6_2 + 1 CARRY4.
module s88m_bw62 (
    input  wire [5:0] x,
    input  wire [1:0] y,
    input  wire       high_n,
    output wire [7:0] biased_prod
);

wire       cin;
wire [3:0] gen;
wire [3:0] prop;
wire [3:0] cout;
wire [3:0] sum;

assign gen[3]  = 1'b0;
assign prop[3] = high_n;

LUT6_2 #(
    .INIT(64'h6ac06ac0a0a0a0a0)
) low01_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[0]), .I3(x[1]),
    .I4(1'b1), .I5(1'b1),
    .O6(biased_prod[1]), .O5(biased_prod[0])
);

// O5 is product bit 2; O6 is the carry entering column 3.
LUT6_2 #(
    .INIT(64'hbbbf333f266a8cc0)
) low2_cin_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[0]), .I3(x[1]),
    .I4(x[2]), .I5(1'b1),
    .O6(cin), .O5(biased_prod[2])
);

// For k=3..5:
//   positive = x[k]   & y[0]
//   negative = ~(x[k-1] & y[1])
//   prop = positive ^ negative, gen = positive & negative.
LUT6_2 #(
    .INIT(64'h2a002a00953f953f)
) gp3_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[2]), .I3(x[3]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[0]), .O5(prop[0])
);

LUT6_2 #(
    .INIT(64'h2a002a00953f953f)
) gp4_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[3]), .I3(x[4]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[1]), .O5(prop[1])
);

LUT6_2 #(
    .INIT(64'h2a002a00953f953f)
) gp5_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[4]), .I3(x[5]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[2]), .O5(prop[2])
);

CARRY4 product_carry (
    .CO(cout), .O(sum),
    .CI(cin), .CYINIT(1'b0),
    .DI(gen), .S(prop)
);

assign biased_prod[6:3] = sum;
assign biased_prod[7]   = cout[3];

endmodule
