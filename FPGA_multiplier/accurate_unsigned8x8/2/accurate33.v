module accurate33 (
    input wire [2:0] a,
    input wire [2:0] b, 

    output wire [5:0] prod  
);


LUT6_2 #(
.INIT(64'h6ac06ac0a0a0a0a0)
) LUT6_inst1 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[0]), 
.I3(a[1]), 
.I4(1'd1), 
.I5(1'd1),

.O6(prod[1]),
.O5(prod[0])
);

LUT6 #(
.INIT(64'h1e665aaab4ccf000)
) LUT6_inst2 (

.I0(b[0]), 
.I1(b[1]), 
.I2(b[2]), 
.I3(a[0]), 
.I4(a[1]), 
.I5(a[2]),

.O(prod[2])
);


LUT6 #(
.INIT(64'h54b46ccc38f00000)
) LUT6_inst3 (

.I0(b[0]), 
.I1(b[1]), 
.I2(b[2]), 
.I3(a[0]), 
.I4(a[1]), 
.I5(a[2]),

.O(prod[3])
);

LUT6 #(
.INIT(64'h983870f0c0000000)
) LUT6_inst4 (

.I0(b[0]), 
.I1(b[1]), 
.I2(b[2]), 
.I3(a[0]), 
.I4(a[1]), 
.I5(a[2]),

.O(prod[4])
);

LUT6 #(
.INIT(64'he0c0800000000000)
) LUT6_inst5 (

.I0(b[0]), 
.I1(b[1]), 
.I2(b[2]), 
.I3(a[0]), 
.I4(a[1]), 
.I5(a[2]),

.O(prod[5])
);


endmodule