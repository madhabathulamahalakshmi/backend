from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import base64
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models
class CaptionRequest(BaseModel):
    image_base64: str
    style: str  # funny, motivational, professional, aesthetic
    auto_hashtags: bool = True

class CaptionResponse(BaseModel):
    id: str
    caption: str
    hashtags: List[str]
    style: str
    image_base64: str
    created_at: datetime

class FavoriteCaption(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    caption: str
    hashtags: List[str]
    style: str
    image_base64: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Settings(BaseModel):
    id: str = Field(default="user_settings")
    dark_mode: bool = False
    auto_hashtags: bool = True

# AI Caption Generation Service
class CaptionAIService:
    def __init__(self):
        self.api_key = os.getenv("EMERGENT_LLM_KEY")
        if not self.api_key:
            raise ValueError("EMERGENT_LLM_KEY not found in environment variables")
        
    async def generate_caption(self, image_base64: str, style: str, auto_hashtags: bool = True) -> dict:
        try:
            # Import the LLM integration library
            from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
            
            # Style-specific prompts
            style_prompts = {
                "funny": "Create a funny, witty, and humorous caption for this image. Be playful, use puns if appropriate, and make it entertaining. Keep it lighthearted and clever.",
                "motivational": "Create an inspiring and motivational caption for this image. Focus on positive energy, encouragement, growth mindset, and empowerment. Make it uplifting and inspiring.",
                "professional": "Create a professional, business-appropriate caption for this image. Keep it polished, sophisticated, and suitable for LinkedIn or corporate settings. Be concise and impactful.",
                "aesthetic": "Create a beautiful, artistic, and aesthetic caption for this image. Focus on mood, atmosphere, visual elements, and poetic language. Make it dreamy and visually appealing."
            }
            
            base_prompt = style_prompts.get(style, style_prompts["aesthetic"])
            
            hashtag_instruction = ""
            if auto_hashtags:
                hashtag_instruction = "\n\nAlso provide 5-8 relevant hashtags separated by commas at the end after the caption. Format: Caption text\n\nHashtags: #tag1, #tag2, #tag3"
            
            full_prompt = f"{base_prompt}{hashtag_instruction}\n\nKeep the caption under 150 characters for social media compatibility."
            
            # Initialize chat with GPT-4o for image analysis
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"caption_{uuid.uuid4()}",
                system_message="You are a creative social media caption generator. You analyze images and create engaging captions in different styles."
            ).with_model("openai", "gpt-4o")
            
            # Create image content from base64
            image_content = ImageContent(image_base64=image_base64)
            
            # Create user message with image
            user_message = UserMessage(
                text=full_prompt,
                file_contents=[image_content]
            )
            
            # Generate caption
            response = await chat.send_message(user_message)
            
            # Parse response
            caption_text = response.strip()
            hashtags = []
            
            if auto_hashtags and "Hashtags:" in caption_text:
                parts = caption_text.split("Hashtags:")
                caption_text = parts[0].strip()
                hashtag_text = parts[1].strip()
                hashtags = [tag.strip() for tag in hashtag_text.replace('#', '').split(',') if tag.strip()]
            
            return {
                "caption": caption_text,
                "hashtags": hashtags
            }
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            # Fallback captions if AI fails
            fallback_captions = {
                "funny": "When life gives you moments, capture them! 📸",
                "motivational": "Every moment is a new beginning. ✨",
                "professional": "Excellence in every detail.",
                "aesthetic": "Beauty in the everyday moments."
            }
            return {
                "caption": fallback_captions.get(style, "Capturing life's beautiful moments."),
                "hashtags": ["life", "moments", "photography", "memories"] if auto_hashtags else []
            }

# Initialize AI service
try:
    ai_service = CaptionAIService()
except Exception as e:
    print(f"Warning: Could not initialize AI service: {e}")
    ai_service = None

# Routes
@api_router.post("/generate-caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    try:
        # Check if AI service is available
        if not ai_service:
            # Fallback captions if AI service is not available
            fallback_captions = {
                "funny": "When life gives you moments, capture them! 📸",
                "motivational": "Every moment is a new beginning. ✨",
                "professional": "Excellence in every detail.",
                "aesthetic": "Beauty in the everyday moments."
            }
            ai_result = {
                "caption": fallback_captions.get(request.style, "Capturing life's beautiful moments."),
                "hashtags": ["life", "moments", "photography", "memories"] if request.auto_hashtags else []
            }
        else:
            # Generate AI caption
            ai_result = await ai_service.generate_caption(
                request.image_base64, 
                request.style, 
                request.auto_hashtags
            )
        
        # Create response
        caption_response = CaptionResponse(
            id=str(uuid.uuid4()),
            caption=ai_result["caption"],
            hashtags=ai_result["hashtags"],
            style=request.style,
            image_base64=request.image_base64,
            created_at=datetime.utcnow()
        )
        
        # Save to history
        await db.caption_history.insert_one(caption_response.dict())
        
        return caption_response
        
    except Exception as e:
        logging.error(f"Error in generate_caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")

@api_router.post("/favorites")
async def add_favorite(favorite: FavoriteCaption):
    try:
        await db.favorites.insert_one(favorite.dict())
        return {"message": "Added to favorites"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/favorites", response_model=List[FavoriteCaption])
async def get_favorites():
    try:
        favorites = await db.favorites.find().sort("created_at", -1).to_list(100)
        return [FavoriteCaption(**fav) for fav in favorites]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/favorites/{favorite_id}")
async def delete_favorite(favorite_id: str):
    try:
        result = await db.favorites.delete_one({"id": favorite_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Favorite not found")
        return {"message": "Favorite deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/history", response_model=List[CaptionResponse])
async def get_history():
    try:
        history = await db.caption_history.find().sort("created_at", -1).limit(10).to_list(10)
        return [CaptionResponse(**item) for item in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/settings", response_model=Settings)
async def get_settings():
    try:
        settings = await db.settings.find_one({"id": "user_settings"})
        if not settings:
            # Create default settings
            default_settings = Settings()
            await db.settings.insert_one(default_settings.dict())
            return default_settings
        return Settings(**settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.put("/settings")
async def update_settings(settings: Settings):
    try:
        await db.settings.update_one(
            {"id": "user_settings"},
            {"$set": settings.dict()},
            upsert=True
        )
        return {"message": "Settings updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/")
async def root():
    return {"message": "Capisiri API is running!"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


