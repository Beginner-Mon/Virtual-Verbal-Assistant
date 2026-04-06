import asyncio
import sys
import time
import logging
logging.getLogger().setLevel(logging.CRITICAL)
sys.path.append('.')
from agents.semantic_bridge import SemanticBridgeService

async def main():
    s = SemanticBridgeService()
    t0 = time.time()
    res = await s.translate_async("Show me how to do a glute bridge")
    print(f"\nRESULT: '{res}'")
    print(f"Time: {time.time()-t0:.2f}s")
    
if __name__ == '__main__':
    asyncio.run(main())
