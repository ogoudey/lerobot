from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio

from dataclasses import dataclass, asdict

MODEL="o3-mini"

@dataclass
class Synonym(BaseModel):
    synonym: str

class Synonym_Agent:
    def __init__(self):
        self.agent = Agent(
            name="Synonym Agent",
            instructions="You take a task and output a synonym for the task that means the same thing. For example, 'put the cube in the bowl' => 'pick up the cube and place it in the bowl'. Try not to add any new content. Be very creative and poetic though. Also, with some probability, say, 0.5, make no change at all.",
            model=MODEL,   # cheapest currently available model
            output_type=Synonym,
        )
    
    async def main(self, prompt):
        
        result = await Runner.run(self.agent, prompt)
        synonym = result.final_output.synonym
        print(prompt, "->", synonym)
        return synonym
    
    def synonomize(self, prompt):
        return asyncio.run(self.main(prompt))
    
