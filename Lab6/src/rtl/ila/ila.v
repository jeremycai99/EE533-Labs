/* file: ila.v
 * Description: Minimal debug ILA — clock gating + live probe readout.
 *
 * Refined Logic:
 * - soc_start defaults to 1 (High) so CPU runs immediately on power-up.
 * - Bit 3 of CTRL register is now a Level control, not a pulse.
 *
 * Register Map (3-bit address):
 *
 *   Addr  Name       R/W  Description
 *   ───── ────────── ──── ─────────────────────────────────────
 *   0x0   CTRL        W   [0] step   [1] run    [2] stop
 *                          [3] cpu_run (Level: 1=Run, 0=Stop/Reset)
 *                          [4] (reserved)
 *                          [5] clear
 *   0x1   STATUS      R   [0] stopped    [1] running
 *                          [2] stepping   [3] soc_busy
 *                          [4] cpu_running [5] debug_mode
 *   0x2   PROBE_SEL  RW   [4:0] debug mux select
 *   0x3   PROBE       R   Live cpu_debug_data
 *   0x4   CYCLE_CNT   R   32-bit cycle counter
 *
 * Author: Jeremy Cai
 * Date: Feb. 14, 2026
 * Version: 
 */

`ifndef ILA_V
`define ILA_V

`include "define.v"

module ila (
    input  wire clk,
    input  wire rst_n,

    // Register interface
    input  wire [2:0] ila_addr,
    input  wire [`MMIO_DATA_WIDTH-1:0] ila_din,
    input  wire ila_we,
    output reg  [`MMIO_DATA_WIDTH-1:0] ila_dout,

    // CPU debug probe
    output wire [4:0] cpu_debug_sel,
    input  wire [`DATA_WIDTH-1:0] cpu_debug_data,

    // CPU start control
    output wire soc_start,

    // Clock gating
    input  wire debug_mode,
    output wire soc_clk_en,

    // soc_driver busy — keep clock alive during its transactions
    input  wire soc_busy,

    input wire txn_pending
);

    // Cycle Counter
    reg [31:0] cycle_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cycle_cnt <= 32'd0;
        else        cycle_cnt <= cycle_cnt + 32'd1;
    end

    // Register Write Logic
    reg step_pulse, run_pulse, stop_pulse;
    reg clear_pulse;
    reg [4:0] probe_sel_r;
    
    // CPU Control Register (Level based)
    reg cpu_run_level; 

    assign cpu_debug_sel = probe_sel_r;

    // Output is directly driven by the register level
    assign soc_start  = cpu_run_level;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            step_pulse    <= 1'b0;
            run_pulse     <= 1'b0;
            stop_pulse    <= 1'b0;
            clear_pulse   <= 1'b0;
            probe_sel_r   <= 5'd0;
            
            // NOTE: Defaulting to 0 means CPU is stopped initially.
            // If you need to start the CPU immediately, you must write 1 
            // to this register first to unblock the SoC req_rdy signal.
            cpu_run_level <= 1'b0; 
        end else begin
            // Pulses auto-clear every cycle unless written
            step_pulse    <= 1'b0;
            run_pulse     <= 1'b0;
            stop_pulse    <= 1'b0;
            clear_pulse   <= 1'b0;

            if (ila_we) begin
                case (ila_addr)
                    3'h0: begin // CTRL Register
                        step_pulse    <= ila_din[0];
                        run_pulse     <= ila_din[1];
                        stop_pulse    <= ila_din[2];
                        cpu_run_level <= ila_din[3]; // Level set
                        clear_pulse   <= ila_din[5];
                    end
                    3'h2: begin // PROBE_SEL
                        probe_sel_r   <= ila_din[4:0];
                    end
                    default: ;
                endcase
            end
        end
    end

    // Clock Gating Control Logic
    reg run_mode;
    reg step_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_mode    <= 1'b0;
            step_active <= 1'b0;
        end else if (clear_pulse) begin
            run_mode    <= 1'b0;
            step_active <= 1'b0;
            cpu_run_level <= 1'b0; // Ensure CPU is stopped on clear
        end else begin
            // Step Logic: Active for exactly one cycle when requested
            if (step_active)
                step_active <= 1'b0;
            else if (step_pulse && !run_mode)
                step_active <= 1'b1;

            // Run/Stop Logic
            if (run_pulse)  run_mode <= 1'b1;
            if (stop_pulse) run_mode <= 1'b0;
        end
    end

    // Clock Enable Logic
    // Enabled if:
    // 1. Not in debug mode (normal operation)
    // 2. In debug mode AND (Running OR Stepping OR External Bus Busy)
    assign soc_clk_en = (!debug_mode) || (run_mode || step_active || soc_busy || txn_pending);

    // Register Read Logic
    always @(*) begin
        case (ila_addr)
            3'h0:    ila_dout = {{(`MMIO_DATA_WIDTH-6){1'b0}}, 
                                  clear_pulse, 
                                  1'b0, // reserved
                                  cpu_run_level, 
                                  stop_pulse, 
                                  run_pulse, 
                                  step_pulse};
            3'h1:    ila_dout = {{(`MMIO_DATA_WIDTH-6){1'b0}},
                                  debug_mode,
                                  cpu_run_level,
                                  soc_busy,
                                  step_active,
                                  run_mode,
                                  (debug_mode & ~run_mode & ~step_active)}; // Stopped status
            3'h2:    ila_dout = {{(`MMIO_DATA_WIDTH-5){1'b0}}, probe_sel_r};
            3'h3:    ila_dout = {{(`MMIO_DATA_WIDTH-`DATA_WIDTH){1'b0}}, cpu_debug_data};
            3'h4:    ila_dout = {{(`MMIO_DATA_WIDTH-32){1'b0}}, cycle_cnt};
            default: ila_dout = {`MMIO_DATA_WIDTH{1'b0}};
        endcase
    end

endmodule

`endif // ILA_V