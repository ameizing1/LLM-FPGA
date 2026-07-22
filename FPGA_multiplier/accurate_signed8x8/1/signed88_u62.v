// 6x2 unsigned LUT/carry-chain multiplier used by the unsigned LL block.
// Resources: 5 LUT6_2 + 1 CARRY4.
module signed88_u62 (
    input  wire [5:0] a,
    input  wire [1:0] b,
    input  wire       high,
    output wire [7:0] prod
);

wire       cin;
wire [3:0] gen;
wire [3:0] prop;
wire [3:0] cout;
wire [3:0] sum;

assign gen[3]  = 1'b0;
assign prop[3] = high; // a[5] & b[1], shared by the parent LUT6_2

LUT6_2 #(
    .INIT(64'h6ac06ac0a0a0a0a0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O6(prod[1]), .O5(prod[0])
);

LUT6_2 #(
    .INIT(64'h88008000e6aa4c00)
) low2_cin_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(a[2]), .I5(1'b1),
    .O6(cin), .O5(prod[2])
);

LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp3_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[2]), .I3(a[3]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[0]), .O5(prop[0])
);

LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp4_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[1]), .O5(prop[1])
);

LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp5_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]),
    .I4(1'b1), .I5(1'b1),
    .O6(gen[2]), .O5(prop[2])
);

CARRY4 product_carry (
    .CO(cout),
    .O(sum),
    .CI(cin),
    .CYINIT(1'b0),
    .DI(gen),
    .S(prop)
);

assign prod[6:3] = sum;
assign prod[7]   = cout[3];

endmodule
