module comp66 (
    input wire [7:0] plow,
    input wire [7:0] pmid,
    input wire [7:0] phigh,

    output wire [11:0] prod
);

//--------------------
// signal assignment
wire [7:0] a_reg;
wire [5:0] b_reg;
wire [5:0] c_reg;

assign a_reg = phigh;
assign b_reg = pmid[7:2];
assign c_reg = {2'b00,plow[7:4]};


assign prod[1:0] = plow[1:0];

wire cin,aban;

wire [7:0] p,g;
assign p[7] = a_reg[7];
assign g[7:6] = 2'b00;

LUT6_2 #(
.INIT(64'h936c936c5a5a5a5a) 
) LUT6_2_inst1 (

.I0(plow[2]),
.I1(plow[3]), 
.I2(pmid[0]), 
.I3(pmid[1]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(prod[3]), 
.O5(prod[2]) 
);

LUT6_2 #(
.INIT(64'h00000000ec80ec80) 
) LUT6_2_inst2 (

.I0(plow[2]),
.I1(plow[3]), 
.I2(pmid[0]), 
.I3(pmid[1]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(aban), 
.O5(cin) 
);


LUT6_2 #(
.INIT(64'h96969696E8E8E8E8) 
) LUT6_2_inst3 (

.I0(c_reg[0]),
.I1(b_reg[0]), 
.I2(a_reg[0]), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(p[0]), 
.O5(g[0]) 
);


//generate the propagates for each bit for the carry chains
genvar i;
for(i = 1;i < 6;i = i+1) begin:GP

LUT6_2 #(
.INIT(64'h69966996E8E8E8E8) 
) LUT6_2_inst4 (

.I0(c_reg[i]),
.I1(b_reg[i]), 
.I2(a_reg[i]), 
.I3(g[i-1]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(p[i]), 
.O5(g[i]) 
);
end

wire g_aban;

LUT6_2 #(
.INIT(64'h6666666600000000) 
) LUT6_2_inst5 (

.I0(a_reg[6]),
.I1(g[5]), 
.I2(1'd1), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(p[6]), 
.O5(g_aban) 
);


//
//carry chain
//
wire [8:0] c_i;
wire [7:0] sum;
assign c_i[0] = cin;
 

CARRY4 CARRY4_inst1 (
.CO(c_i[4:1]), 
.O(sum[3:0]), 
 
.CI(c_i[0]), 
.CYINIT(1'd0), 
.DI({g[2:0],1'b0}),
.S(p[3:0]) 
);


CARRY4 CARRY4_inst2 (
.CO(c_i[8:5]), 
.O(sum[7:4]),

.CI(c_i[4]), 
.CYINIT(1'd0), 
.DI(g[6:3]), 
.S(p[7:4]) 
);

assign prod[11:4] = sum[7:0];

endmodule