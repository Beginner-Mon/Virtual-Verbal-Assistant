import asyncio
import sys
from mcp import ClientSession
from mcp.client.sse import sse_client

# ĐIỀN ĐỊA CHỈ IP CỦA TASK VÀO ĐÂY (VÍ DỤ: 54.123.45.67)
# Nếu bạn chạy script này trực tiếp trên máy EC2, hãy để là "localhost" hoặc "127.0.0.1"
SERVER_IP = "ĐIỀN_IP_VÀO_ĐÂY"
SERVER_URL = f"http://{SERVER_IP}:8000/sse"

async def run_test():
    if SERVER_IP == "ĐIỀN_IP_VÀO_ĐÂY":
        print("❌ LỖI: Bạn chưa điền địa chỉ IP của Task vào file code!")
        print("Hãy mở file test_kimodo.py và sửa dòng SERVER_IP = '...'")
        sys.exit(1)

    print(f"🔄 Đang kết nối tới Kimodo MCP Server tại: {SERVER_URL}...")
    
    try:
        async with sse_client(SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # 1. Khởi tạo kết nối (Bắt buộc)
                await session.initialize()
                print("✅ Kết nối thành công!\n")
                
                # 2. Liệt kê các công cụ (Tools) có sẵn
                print("🛠️ DANH SÁCH CÔNG CỤ (TOOLS):")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  - {tool.name}")
                print("-" * 40)
                
                # 3. Test công cụ health_check
                print("\n🏥 ĐANG KIỂM TRA SỨC KHỎE SERVER (health_check)...")
                health_result = await session.call_tool("health_check", arguments={})
                print(f"Kết quả:\n{health_result.content[0].text}")
                print("-" * 40)

                # 4. Test tạo chuyển động (generate_motion)
                prompt = "A person waves their hand."
                print(f"\n🏃‍♂️ ĐANG YÊU CẦU AI TẠO CHUYỂN ĐỘNG VỚI PROMPT: '{prompt}'...")
                print("Đang chờ GPU xử lý (có thể mất vài chục giây)...")
                
                motion_result = await session.call_tool("generate_motion", arguments={"prompt": prompt})
                print(f"\n✅ HOÀN THÀNH! Kết quả trả về:\n{motion_result.content[0].text}")
                
    except Exception as e:
        print(f"\n❌ Kết nối thất bại: {str(e)}")
        print("Gợi ý: Hãy kiểm tra lại IP, hoặc đảm bảo Security Group đang mở Port 8000")

if __name__ == "__main__":
    asyncio.run(run_test())
