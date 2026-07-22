module accurate66 (
    input wire [5:0] a,
    input wire [5:0] b, 
    input wire high_1,
    input wire high_2,
    input wire high_3,

    output wire [11:0] prod  
);


wire [7:0] plow,pmid,phigh;

accurate62 Low(
    .a(a),
    .b(b[1:0]),
    .a_high(high_1),
    .prod(plow)
);

accurate62 Mid(
    .a(a),
    .b(b[3:2]),
    .a_high(high_2),
    .prod(pmid)    
);

accurate62 High(
    .a(a),
    .b(b[5:4]),
    .a_high(high_3),
    .prod(phigh)    
);

comp66 comp(
    .plow(plow),
    .pmid(pmid),
    .phigh(phigh),

    .prod(prod)
);

endmodule