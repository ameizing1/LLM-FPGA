// Signed int8 wrapper for an unsigned approx88_cascade core.
//
// This module is a baseline RTL wrapper:
//   signed int8 a,b -> abs magnitudes -> unsigned approx88_cascade -> restore sign.
//
// The output is 17-bit signed to avoid overflow if an approximate unsigned core
// emits a magnitude larger than 32767 before sign restoration.

module signed_approx88_wrapper(
    input wire signed [7:0] a,
    input wire signed [7:0] b,
    output wire signed [16:0] prod
);
    wire sign;
    wire [7:0] mag_a;
    wire [7:0] mag_b;
    wire [15:0] mag_prod;
    wire signed [16:0] mag_prod_ext;

    assign sign = a[7] ^ b[7];
    assign mag_a = a[7] ? (~a + 8'd1) : a;
    assign mag_b = b[7] ? (~b + 8'd1) : b;

    approx88_cascade u_unsigned_core(
        .a(mag_a),
        .b(mag_b),
        .prod(mag_prod)
    );

    assign mag_prod_ext = {1'b0, mag_prod};
    assign prod = sign ? -mag_prod_ext : mag_prod_ext;
endmodule

