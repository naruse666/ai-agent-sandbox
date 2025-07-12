package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/tools"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// llm instance
	llm, err := openai.New(openai.WithModel("gpt-4o"))
	if err != nil {
		log.Fatal(err)
	}

	// tools
	calc := tools.Calculator{}
	toolset := []tools.Tool{calc}
	toolMap := map[string]tools.Tool{calc.Name(): calc}

	// memory
	memory := memory.NewConversationBuffer()

	// one-shot agent
	agent := agents.NewOneShotAgent(
		llm,
		toolset,
		agents.WithMemory(memory),
	)

	// manual logic
	userInput := "What is the result of 3 + 5, and then multiply it by 2?"
	inputs := map[string]string{"input": userInput}
	var steps []schema.AgentStep

	for i := 0; i < 5; i++ {
		actions, finish, err := agent.Plan(ctx, steps, inputs)
		if err != nil {
			log.Fatal(err)
		}

		// if finish is not nil, the agent has completed its task
		if finish != nil {
			fmt.Println("Finished =====")
			fmt.Println("Agent finished with result:", finish.Log)
			memory.SaveContext(
				ctx,
				map[string]any{"input": inputs},
				map[string]any{"output": finish.Log},
			)
			return
		}
		fmt.Printf("Step %d =====\n", i+1)
		fmt.Printf("Plan: %v\n", actions)

		act := actions[0]
		tool, ok := toolMap[act.Tool]
		if !ok {
			log.Fatalf("Tool %s not found", act.Tool)
		}
		fmt.Printf("Planning action: %s with input: %s\n", act.Tool, act.ToolInput)

		// execute tool
		observation, err := tool.Call(ctx, act.ToolInput)
		if err != nil {
			observation = fmt.Sprintf("Error executing tool: %v", err)
		}
		fmt.Printf("Action: %s, Observation: %s\n", act.Tool, observation)

		// update history
		steps = append(steps, schema.AgentStep{
			Action:      act,
			Observation: observation,
		})

		memory.SaveContext(
			ctx,
			map[string]any{"input": act.ToolInput},
			map[string]any{"output": observation},
		)
	}
}
