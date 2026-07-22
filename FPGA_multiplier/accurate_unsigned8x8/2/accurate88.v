module accurate88 (
    input wire [7:0] a,
    input wire [7:0] b, 

    output wire [15:0] prod  
);

wire [1:0] ah,bh;
wire [5:0] al,bl;

wire [11:0] ll;
wire [7:0] hl,lh;
wire [3:0] hh;


assign ah = a[7:6];
assign al = a[5:0];
assign bh = b[7:6];
assign bl = b[5:0];

accurate22 HH(
    .a(ah),
    .b(bh),
    .prod(hh)
);

accurate62 HL(
    .a(al),
    .b(bh),
    .prod(hl)    
);

accurate62 LH(
    .a(bl),
    .b(ah),
    .prod(lh)    
);

accurate66 LL(
    .a(al),
    .b(bl),
    .prod(ll)    
);

comp88 comp(
    .hh(hh),
    .hl(hl),
    .lh(lh),
    .ll(ll),

    .prod(prod)
);

endmodule