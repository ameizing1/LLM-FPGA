// Exact unsigned 3x3 multiplier.
//
// This is the LUT-rich / carry-light leaf used by the 8x8_50 design:
//   1 LUT6_2 produces bits 0 and 1;
//   4 LUT6   directly implement bits 2 through 5.
// Resources: 1 LUT6_2 + 4 LUT6, no CARRY4.
module s88m_mul33 (
    input  wire [2:0] a,
    input  wire [2:0] b,
    output wire [5:0] prod
);

LUT6_2 #(
    .INIT(64'h6ac06ac0a0a0a0a0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O6(prod[1]), .O5(prod[0])
);

LUT6 #(
    .INIT(64'h1e665aaab4ccf000)
) bit2_lut (
    .I0(b[0]), .I1(b[1]), .I2(b[2]),
    .I3(a[0]), .I4(a[1]), .I5(a[2]),
    .O(prod[2])
);

LUT6 #(
    .INIT(64'h54b46ccc38f00000)
) bit3_lut (
    .I0(b[0]), .I1(b[1]), .I2(b[2]),
    .I3(a[0]), .I4(a[1]), .I5(a[2]),
    .O(prod[3])
);

LUT6 #(
    .INIT(64'h983870f0c0000000)
) bit4_lut (
    .I0(b[0]), .I1(b[1]), .I2(b[2]),
    .I3(a[0]), .I4(a[1]), .I5(a[2]),
    .O(prod[4])
);

LUT6 #(
    .INIT(64'he0c0800000000000)
) bit5_lut (
    .I0(b[0]), .I1(b[1]), .I2(b[2]),
    .I3(a[0]), .I4(a[1]), .I5(a[2]),
    .O(prod[5])
);

endmodule
