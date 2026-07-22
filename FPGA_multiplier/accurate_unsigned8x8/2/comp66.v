module comp66 (
    input wire [5:0]hh,
    input wire [5:0]hl,
    input wire [5:0]lh,
    input wire [5:0]ll,

    output wire [11:0] prod
);

//--------------------
// signal assignment
wire [7:0] a_reg;
wire [4:0] b_reg,c_reg;

wire [7:0] p,g;
wire cin;

assign a_reg[1:0] = ll[5:4];
assign a_reg[7:2] = hh[5:0];
assign b_reg[4:0] = lh[5:1];
assign c_reg[4:0] = hl[5:1];


assign prod[2:0] = ll[2:0];
assign p[7:6]=a_reg[7:6];
assign g[7:5]=3'b000;

//-------------
//RCA 
//

LUT6_2 #(
.INIT(64'he8e8e8e896969696) 
) LUT6_2_inst1 (

.I0(ll[3]),
.I1(lh[0]), 
.I2(hl[0]), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(cin), 
.O5(prod[3]) 
);


LUT6_2 #(
.INIT(64'h96969696E8E8E8E8) 
) LUT6_2_inst2 (

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
for(i = 1;i < 5;i = i+1) begin:GP

LUT6_2 #(
.INIT(64'h69966996E8E8E8E8) 
) LUT6_2_inst3 (

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
) LUT6_2_inst4 (

.I0(a_reg[5]),
.I1(g[4]), 
.I2(1'd1), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(p[5]), 
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