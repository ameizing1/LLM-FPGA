// Four-row compressor for an unsigned 6x6 product split into four 3x3 blocks.
// Resources: 7 LUT6_2 + 2 CARRY4.
module s88m_comp66 (
    input  wire [5:0] hh,
    input  wire [5:0] hl,
    input  wire [5:0] lh,
    input  wire [5:0] ll,
    input  wire       shared_x5,
    input  wire       shared_sign,
    output wire [11:0] prod,
    output wire       shared_high_n
);

wire [7:0] a_row;
wire [4:0] b_row;
wire [4:0] c_row;
wire [7:0] p;
wire [7:0] g;
wire       cin;

assign a_row[1:0] = ll[5:4];
assign a_row[7:2] = hh;
assign b_row       = lh[5:1];
assign c_row       = hl[5:1];

assign prod[2:0] = ll[2:0];
assign p[7:6]    = a_row[7:6];
assign g[7:5]    = 3'b000;

// Column 3: O5 is the sum bit; O6 is the carry into column 4.
LUT6_2 #(
    .INIT(64'he8e8e8e896969696)
) column3_lut (
    .I0(ll[3]), .I1(lh[0]), .I2(hl[0]),
    .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O6(cin), .O5(prod[3])
);

// First 3:2 compressor column.
LUT6_2 #(
    .INIT(64'h96969696e8e8e8e8)
) csa0_lut (
    .I0(c_row[0]), .I1(b_row[0]), .I2(a_row[0]),
    .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O6(p[0]), .O5(g[0])
);

genvar i;
generate
    for (i = 1; i < 5; i = i + 1) begin : csa_gp
        LUT6_2 #(
            .INIT(64'h69966996e8e8e8e8)
        ) csa_lut (
            .I0(c_row[i]), .I1(b_row[i]), .I2(a_row[i]), .I3(g[i-1]),
            .I4(1'b1), .I5(1'b1),
            .O6(p[i]), .O5(g[i])
        );
    end
endgenerate

// Tail propagate on O6.  O5 would otherwise be unused, so it also creates one
// signed cross block's final complemented row bit:
//   shared_high_n = ~(shared_x5 & shared_sign)
LUT6_2 #(
    .INIT(64'h0ff00ff077777777)
) tail_lut (
    .I0(shared_x5), .I1(shared_sign),
    .I2(a_row[5]), .I3(g[4]), .I4(1'b1), .I5(1'b1),
    .O6(p[5]), .O5(shared_high_n)
);

wire [8:0] carry;
wire [7:0] sum;
assign carry[0] = cin;

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

assign prod[11:4] = sum;

endmodule
