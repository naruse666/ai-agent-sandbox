from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

app = BedrockAgentCoreApp()
bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0", region_name="us-east-1")

agent = Agent(model=bedrock_model)
print(agent.model.config)


@app.entrypoint
def invoke(payload):
    """Your AI agent function"""
    try:
        user_message = payload.get(
            "prompt", "Hello! How can I help you today?")
        result = agent(user_message)
        return {"result": result.message}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
