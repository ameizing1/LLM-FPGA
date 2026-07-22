module comp88 (
    input wire [4:0]hh,
    input wire [7:0]hl,
    input wire [7:0]lh,
    input wire [11:0]ll,

    output wire [15:0] prod
);

//--------------------
// signal assignment
wire [8:0] a_reg;
wire [7:0] b_reg,c_reg;




assign a_reg[5:0] = ll[11:6];
assign a_reg[8:6] = hh[2:0]; //hh = {a[7],b[7],hh_2,hh_1,hh_0}
assign b_reg[7:0] = lh[7:0];
assign c_reg[7:0] = hl[7:0];

assign prod[5:0] = ll[5:0];


wire [11:0] p,g;
assign g[11:9] = 3'b000;
assign p[11:9] = {2'b00,g[8]};

//-------------
//RCA 
//


LUT6_2 #(
.INIT(64'h96969696E8E8E8E8) 
) LUT6_2_inst1 (

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
for(i = 1;i < 8;i = i+1) begin:GP 

LUT6_2 #(
.INIT(64'h69966996E8E8E8E8) 
) LUT6_2_inst2 (

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

LUT6_2 #(
.INIT(64'h96669666a000a000) 
) LUT6_2_inst3 (

.I0(a_reg[8]),
.I1(g[7]), 
.I2(hh[4]), 
.I3(hh[3]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(p[8]), 
.O5(g[8])
);


//
//carry chain
//
wire [12:0] c_i;
wire [11:0] sum;
assign c_i[0] = 1'b0;
 

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

CARRY4 CARRY4_inst3 (
.CO(c_i[12:9]), 
.O(sum[11:8]),

.CI(c_i[8]), 
.CYINIT(1'd0), 
.DI(g[10:7]), 
.S(p[11:8]) 
);

assign prod[15:6] = sum[9:0];

endmodule