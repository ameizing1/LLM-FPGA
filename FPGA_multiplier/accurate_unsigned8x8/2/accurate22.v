module accurate22 (
    input wire [1:0] a,
    input wire [1:0] b,
    output wire [3:0] prod
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


LUT6_2 #(
.INIT(64'h800080004c004c00)
) LUT6_inst2 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[0]), 
.I3(a[1]), 
.I4(1'd1), 
.I5(1'd1),

.O6(prod[3]),
.O5(prod[2])
);

endmodule