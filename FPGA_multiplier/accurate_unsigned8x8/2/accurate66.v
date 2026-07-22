module accurate66 (
    input wire [5:0] a,
    input wire [5:0] b, 

    output wire [11:0] prod  
);

wire [2:0] ah,al,bh,bl;

wire [5:0] hh,hl,lh,ll;

assign ah = a[5:3];
assign al = a[2:0];
assign bh = b[5:3];
assign bl = b[2:0];

accurate33 HH(
    .a(ah),
    .b(bh),
    .prod(hh)
);

accurate33 HL(
    .a(ah),
    .b(bl),
    .prod(hl)    
);

accurate33 LH(
    .a(al),
    .b(bh),
    .prod(lh)    
);

accurate33 LL(
    .a(al),
    .b(bl),
    .prod(ll)    
);

comp66 comp(
    .hh(hh),
    .hl(hl),
    .lh(lh),
    .ll(ll),

    .prod(prod)
);

endmodule