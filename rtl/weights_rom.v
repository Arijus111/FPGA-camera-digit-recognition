module weights_rom (input clk,
                    input [15:0] addr,
                    output reg [7:0] data);
    
    reg [7:0] weights [0:50175]; //weights array
    
    initial begin
        $readmemh("w1.mem", weights);
    end
    
    always@(posedge clk) begin
        data <= weights [addr];
    end
endmodule
