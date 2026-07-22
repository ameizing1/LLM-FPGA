module accurate88 (
    input wire [7:0] a,
    input wire [7:0] b, 

    output wire [15:0] prod  
);

wire [1:0] ah,bh;
wire [5:0] al,bl;

wire [11:0] ll;
wire [7:0] hl,lh;
wire hh_0,hh_1,hh_2;


assign ah = a[7:6];
assign al = a[5:0];
assign bh = b[7:6];
assign bl = b[5:0];

wire ll_high1,ll_high2,ll_high3;
wire hl_high,lh_high;

LUT6_2 #(
.INIT(64'ha0a0a0a088888888) 
) LUT6_2_inst1 (

.I0(a[5]),
.I1(b[1]), 
.I2(b[3]), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(ll_high2), 
.O5(ll_high1) 
);

LUT6_2 #(
.INIT(64'hc0c0c0c088888888) 
) LUT6_2_inst2 (

.I0(a[5]),
.I1(b[5]), 
.I2(a[7]), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1), 

.O6(hl_high), 
.O5(ll_high3) 
);

LUT6_2 #(
.INIT(64'hf000f00088888888) 
) LUT6_2_inst3 (

.I0(a[5]),
.I1(b[7]), 
.I2(a[6]), 
.I3(b[6]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(hh_0), 
.O5(lh_high) 
);

LUT6_2 #(
.INIT(64'h8000800078887888) 
) LUT6_2_inst4 (

.I0(a[7]),
.I1(b[6]), 
.I2(a[6]), 
.I3(b[7]), 
.I4(1'd1), 
.I5(1'd1), 

.O6(hh_2), 
.O5(hh_1) 
);

accurate66 LL(
    .a(al),
    .b(bl),
    .high_1(ll_high1),
    .high_2(ll_high2),
    .high_3(ll_high3), 
    .prod(ll)    
);

accurate62 HL(
    .a(bl),
    .b(ah),
    .a_high(hl_high),
    .prod(hl)    
);

accurate62 LH(
    .a(al),
    .b(bh),
    .a_high(lh_high),
    .prod(lh)    
);

wire [4:0] hh;
assign hh = {a[7],b[7],hh_2,hh_1,hh_0};

comp88 comp(
    .hh(hh),
    .hl(hl),
    .lh(lh),
    .ll(ll),

    .prod(prod)
);

endmodule