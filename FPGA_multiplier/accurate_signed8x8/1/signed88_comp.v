// Final signed 8x8 compressor.
//
// hh = {a_sign, b_sign, hh2_bw, hh1, hh0}
// hl_bias/lh_bias are signed cross products with a fixed +128 bias.
// The modified hh2 and complemented g[8] absorb both cross-product biases
// and the final Baugh-Wooley correction without another adder.
// Resources: 9 LUT6_2 + 3 CARRY4.
module signed88_comp (
    input  wire [4:0]  hh,
    input  wire [7:0]  hl_bias,
    input  wire [7:0]  lh_bias,
    input  wire [11:0] ll,
    output wire [15:0] prod
);

wire [8:0] a_row;
wire [7:0] b_row;
wire [7:0] c_row;

assign a_row[5:0] = ll[11:6];
assign a_row[8:6] = hh[2:0];
assign b_row       = lh_bias;
assign c_row       = hl_bias;
assign prod[5:0]   = ll[5:0];

wire [11:0] p;
wire [11:0] g;

assign g[11:9] = 3'b000;
assign p[11:9] = {2'b00, g[8]};

// Column 0 of the three-row compressor.
LUT6_2 #(
    .INIT(64'h96969696e8e8e8e8)
) csa0_lut (
    .I0(c_row[0]), .I1(b_row[0]), .I2(a_row[0]), .I3(1'b1),
    .I4(1'b1), .I5(1'b1),
    .O6(p[0]), .O5(g[0])
);

// p[i] includes the saved carry g[i-1]; g[i] remains the raw majority.
genvar i;
generate
    for (i = 1; i < 8; i = i + 1) begin : csa_gp
        LUT6_2 #(
            .INIT(64'h69966996e8e8e8e8)
        ) csa_lut (
            .I0(c_row[i]), .I1(b_row[i]), .I2(a_row[i]), .I3(g[i-1]),
            .I4(1'b1), .I5(1'b1),
            .O6(p[i]), .O5(g[i])
        );
    end
endgenerate

// O6: p8 = hh2_bw ^ (a_sign & b_sign) ^ g7
// O5: g8 = ~(hh2_bw & (a_sign & b_sign))
// The complemented O5 folds the final 16'h8000 BW correction into bit 15.
LUT6_2 #(
    .INIT(64'h966696665fff5fff)
) signed_top_lut (
    .I0(a_row[8]), .I1(g[7]), .I2(hh[4]), .I3(hh[3]),
    .I4(1'b1), .I5(1'b1),
    .O6(p[8]), .O5(g[8])
);

wire [12:0] carry;
wire [11:0] sum;

assign carry[0] = 1'b0;

CARRY4 carry_0 (
    .CO(carry[4:1]), .O(sum[3:0]),
    .CI(carry[0]), .CYINIT(1'b0),
    .DI({g[2:0], 1'b0}), .S(p[3:0])
);

CARRY4 carry_1 (
    .CO(carry[8:5]), .O(sum[7:4]),
    .CI(carry[4]), .CYINIT(1'b0),
    .DI(g[6:3]), .S(p[7:4])
);

CARRY4 carry_2 (
    .CO(carry[12:9]), .O(sum[11:8]),
    .CI(carry[8]), .CYINIT(1'b0),
    .DI(g[10:7]), .S(p[11:8])
);

assign prod[15:6] = sum[9:0];

endmodule
