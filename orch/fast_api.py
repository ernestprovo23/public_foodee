import requests
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import uuid
import jwt
from jose import jwt, JWTError
from passlib.context import CryptContext
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as aioredis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, EmailStr
from typing import List, Optional, Dict, Any, AsyncGenerator
from typing import Callable
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
import os
from typing import List

# Import the auth router
from auth_routes import router as auth_router

# Load environment variables
load_dotenv()

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError(
        "No JWT_SECRET_KEY environment variable found. "
        "Please set JWT_SECRET_KEY with a secure secret key in your .env file."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('foodee_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Foodee.API")


# Update the existing FastAPI initialization to include auth configurations
async def init_redis():
    try:
        redis = await aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
        await FastAPILimiter.init(redis)
        return redis
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        # Initialize Redis
        app.state.redis = await init_redis()
        logger.info("Redis initialized successfully")

        # Initialize MongoDB connection
        app.state.mongodb = MongoClient(MONGO_CONN_STRING)
        logger.info("MongoDB initialized successfully")

        yield

    finally:
        # Shutdown
        if hasattr(app.state, "redis"):
            await app.state.redis.close()
        if hasattr(app.state, "mongodb"):
            app.state.mongodb.close()
        logger.info("Application shutdown complete")


# Update FastAPI app initialization
app = FastAPI(
    title="Foodee API",
    description="API for Foodee meal planning and nutrition service",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the auth router
app.include_router(
    auth_router,
    prefix="/api",
    tags=["authentication"]
)


# Prometheus metrics
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ACTIVE_USERS = Counter('active_users_total', 'Total number of active users')
MEAL_PLANS_GENERATED = Counter('meal_plans_generated_total', 'Total number of meal plans generated')
ERROR_COUNT = Counter('error_count_total', 'Total number of errors', ['type'])


# Rate limiting function
async def rate_limit(request: Request, response: Response, times: int, seconds: int):
    try:
        redis = request.app.state.redis
        key = f"rate_limit:{request.client.host}"
        current = await redis.get(key)

        if current is None:
            await redis.setex(key, seconds, 1)
        else:
            current = int(current)
            if current >= times:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests"
                )
            await redis.incr(key)
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        # If Redis fails, allow the request but log the error
        pass


# MongoDB connection setup
class MongoDB:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGO_DB_CONN_STRING'))
        self.db = self.client.foode

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


class MealPlanRequest(BaseModel):
    username: constr(min_length=1) = Field(..., description="Username for the meal plan")
    num_iterations: int = Field(..., ge=1, le=1000, description="Number of meal plans to generate")


class SessionInfo(BaseModel):
    session_id: str
    timestamp: datetime


class MealPlanStatus(BaseModel):
    username: str
    status: str
    session_ids: List[str]
    timestamp: datetime
    details: Optional[Dict] = None


class UserProfile(BaseModel):
    username: str
    email: EmailStr
    dietary_restrictions: List[str] = []
    food_preferences: Dict[str, List[str]] = {}


class MetricsResponse(BaseModel):
    total_requests: int
    active_users: int
    meal_plans_generated: int
    average_response_time: float
    error_rate: float


# Security Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    scopes: List[str] = []


class UserInDB(User):
    hashed_password: str


# Security configuration
class SecurityConfig:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 30
    PASSWORD_MIN_LENGTH = 8

    # Define available scopes
    SCOPES = {
        "meals:read": "Read meal plans",
        "meals:write": "Create and modify meal plans",
        "profile:read": "Read user profile",
        "profile:write": "Modify user profile",
        "admin": "Administrative access"
    }


# Password and token handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes=SecurityConfig.SCOPES
)


# Password and token utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters long")
    return pwd_context.hash(password)


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    return jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)


def create_refresh_token(data: Dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    return jwt.encode(to_encode, SecurityConfig.SECRET_KEY, algorithm=SecurityConfig.ALGORITHM)


# User authentication and verification
async def get_user(db: MongoClient, username: str) -> Optional[UserInDB]:
    if user_dict := db.users.find_one({"username": username}):
        return UserInDB(**user_dict)
    return None


async def authenticate_user(db: MongoClient, username: str, password: str) -> Optional[UserInDB]:
    user = await get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# Dependencies
def get_db():
    try:
        yield app.state.mongodb.foode
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

# Add error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
        security_scopes: SecurityScopes,
        token: str = Depends(oauth2_scheme),
        db: MongoClient = Depends(get_db)
) -> User:
    authenticate_value = f'Bearer scope="{security_scopes.scope_str}"' if security_scopes.scopes else "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = jwt.decode(token, SecurityConfig.SECRET_KEY, algorithms=[SecurityConfig.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, username=username)
    except JWTError:
        raise credentials_exception

    user = await get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception

    # Verify required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user


async def get_current_active_user(
        current_user: User = Security(get_current_user, scopes=[])
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# FastAPI endpoint implementations
@app.post("/token", response_model=Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: MongoClient = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": form_data.scopes
        },
        expires_delta=access_token_expires
    )

    refresh_token = create_refresh_token(
        data={
            "sub": user.username,
            "scopes": form_data.scopes
        }
    )

    # Store refresh token in database
    db.refresh_tokens.insert_one({
        "username": user.username,
        "refresh_token": refresh_token,
        "created_at": datetime.utcnow()
    })

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@app.post("/token/refresh", response_model=Token)
async def refresh_token(
        refresh_token: str,
        db: MongoClient = Depends(get_db)
):
    try:
        payload = jwt.decode(refresh_token, SecurityConfig.SECRET_KEY, algorithms=[SecurityConfig.ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Verify refresh token exists in database
        stored_token = db.refresh_tokens.find_one({
            "username": username,
            "refresh_token": refresh_token
        })

        if not stored_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        # Create new access token
        access_token_expires = timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username, "scopes": payload.get("scopes", [])},
            expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@app.post("/token/revoke")
async def revoke_token(
        current_user: User = Depends(get_current_active_user),
        db: MongoClient = Depends(get_db)
):
    # Remove refresh tokens for the user
    db.refresh_tokens.delete_many({"username": current_user.username})
    return {"detail": "Tokens revoked successfully"}


# Background task orchestrator
class MealPlanOrchestrator:
    @staticmethod
    async def run_process(username: str, num_iterations: int) -> List[str]:
        try:
            from orch_sng_ml_wc import FoodeeOrchestrator

            orchestrator = FoodeeOrchestrator()
            await orchestrator.orchestrate_full_process(username, num_iterations)

            MEAL_PLANS_GENERATED.inc()

            session_ids = await orchestrator.get_recent_session_ids(username, num_iterations)
            return session_ids

        except Exception as e:
            ERROR_COUNT.labels(type='orchestration').inc()
            logger.error(f"Error in meal plan orchestration: {e}")
            raise


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    return response


def rate_limit_dependency(times: int = 10, seconds: int = 60) -> Callable:
    async def rate_limit(request: Request):
        try:
            redis = request.app.state.redis
            key = f"rate_limit:{request.client.host}"
            current = await redis.get(key)

            if current is None:
                await redis.setex(key, seconds, 1)
            else:
                current = int(current)
                if current >= times:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Too many requests"
                    )
                await redis.incr(key)
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
    return Depends(rate_limit)

# Then update your endpoint to use it like this:
@app.post("/api/meal-plan", response_model=MealPlanStatus)
async def create_meal_plan(
    request: Request,
    meal_plan_request: MealPlanRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: MongoClient = Depends(get_db),
    _=rate_limit_dependency(times=10, seconds=60)
):
    """
    Create a new meal plan for a user.
    Requires authentication and is rate limited.
    """
    try:
        user = db.user_summary.find_one({"username": meal_plan_request.username})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        status_doc = {
            "username": meal_plan_request.username,
            "status": "processing",
            "timestamp": datetime.now(),
            "session_ids": []
        }

        result = db.meal_plan_status.insert_one(status_doc)

        background_tasks.add_task(
            MealPlanOrchestrator.run_process,
            meal_plan_request.username,
            meal_plan_request.num_iterations
        )

        return MealPlanStatus(
            username=meal_plan_request.username,
            status="processing",
            session_ids=[],
            timestamp=datetime.now()
        )

    except Exception as e:
        ERROR_COUNT.labels(type='meal_plan_creation').inc()
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/meal-plan/{username}/status", response_model=MealPlanStatus)
async def get_meal_plan_status(
        username: str,
        current_user: User = Depends(get_current_user),
        db: MongoClient = Depends(get_db)
):
    """
    Get the status of a user's meal plan generation.
    Requires authentication.
    """
    try:
        status = db.meal_plan_status.find_one(
            {"username": username},
            sort=[("timestamp", -1)]
        )

        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No meal plan status found for user"
            )

        return MealPlanStatus(
            username=status["username"],
            status=status["status"],
            session_ids=status["session_ids"],
            timestamp=status["timestamp"],
            details=status.get("details")
        )

    except Exception as e:
        ERROR_COUNT.labels(type='status_check').inc()
        logger.error(f"Error getting meal plan status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/meal-plan/{username}/results/{session_id}")
async def get_meal_plan_results(
    username: str,
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: MongoClient = Depends(get_db),
    _: None = Depends(RateLimiter(times=30, seconds=60))
):
    """
    Get the results for a specific meal plan session.
    Requires authentication and is rate limited.
    """
    try:
        results = {
            "meal_plan": db.results.find_one({
                "username": username,
                "session_id": session_id
            }),
            "shopping_list": db.shopping_lists_with_prices.find_one({
                "username": username,
                "session_id": session_id
            }),
            "nutrition_info": db.recipe_nutrition_info.find_one({
                "username": username,
                "session_id": session_id
            })
        }

        if not any(results.values()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No results found for this session"
            )

        for result in results.values():
            if result and "_id" in result:
                result["_id"] = str(result["_id"])

        return results

    except Exception as e:
        ERROR_COUNT.labels(type='results_retrieval').inc()
        logger.error(f"Error getting meal plan results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/stats", response_model=MetricsResponse)
async def get_stats(current_user: User = Depends(get_current_user)):
    """
    Get API statistics and metrics.
    Requires authentication.
    """
    try:
        total_requests = sum(
            REQUESTS.labels(method, endpoint, status)._value.get()
            for method, endpoint, status in REQUESTS._metrics.keys()
        )
        active_users = ACTIVE_USERS._value.get()
        meal_plans = MEAL_PLANS_GENERATED._value.get()
        avg_response_time = (
            LATENCY._sum.get()
            / max(LATENCY._count.get(), 1)
        )
        error_rate = sum(
            ERROR_COUNT.labels(type=error_type)._value.get()
            for error_type in ERROR_COUNT._metrics.keys()
        ) / max(total_requests, 1)

        return MetricsResponse(
            total_requests=total_requests,
            active_users=active_users,
            meal_plans_generated=meal_plans,
            average_response_time=avg_response_time,
            error_rate=error_rate
        )
    except Exception as e:
        ERROR_COUNT.labels(type='stats').inc()
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(),
            "services": {}
        }

        # Check MongoDB
        try:
            with MongoDB() as db:
                db.command('ping')
                health_status["services"]["mongodb"] = "up"
        except Exception as e:
            health_status["services"]["mongodb"] = f"down: {str(e)}"

        # Check Redis
        try:
            await app.state.redis.ping()
            health_status["services"]["redis"] = "up"
        except Exception as e:
            health_status["services"]["redis"] = f"down: {str(e)}"

        # Overall status
        if all(v == "up" for v in health_status["services"].values()):
            return health_status
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )

    except Exception as e:
        ERROR_COUNT.labels(type='health_check').inc()
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(),
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=1233,
        reload=True,
        log_level="info"
    )