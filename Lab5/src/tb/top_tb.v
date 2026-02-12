`timescale 1ns/1ps

`include "define.v"
`include "top.v"

module top_tb;
reg clk;
reg rst_n;

// Instantiate the top module
top u_top (
    .clk(clk),
    .rst_n(rst_n)
);
// Clock generation: 10ns period (100MHz)
initial begin
    clk = 0;
    forever #5 clk = ~clk; // Toggle clock every 5ns
end
// Reset generation: Assert reset for the first 20ns
initial begin
    rst_n = 0; // Assert reset (active low)
    #20 rst_n = 1; // Deassert reset after 20ns
end

// Simulation runtime control: Run the simulation for a certain time and then finish
initial begin
    #1000; // Run the simulation for 1000ns (1 microsecond)
    $finish; // End the simulation
end

endmodule