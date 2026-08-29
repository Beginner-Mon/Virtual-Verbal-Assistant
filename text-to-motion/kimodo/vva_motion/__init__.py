"""Bảng job motion — dùng chung giữa Lambda vva-agent và worker Kimodo.

Chỉ phụ thuộc boto3 + stdlib. KHÔNG import gì của langgraph_agents vào package này:
nó được COPY vào cả image Kimodo (nơi không có langgraph_agents) lẫn image Lambda.
"""
