# routers/audio_processing.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
import os
import time
import logging
import shutil
import json
from database import SessionLocal
from utils.audio_processing import isolate_voice, get_available_models
from utils.audio_conversion import convert_audio_to_wav, is_valid_wav

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Directory setup
TEMP_DIR = "data/temp"
MIXED_DIR = "data/mixed"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MIXED_DIR, exist_ok=True)

@router.get("/models")
async def get_models():
    """Get all available models for the frontend."""
    try:
        available_models = get_available_models()
        return available_models
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@router.post("/process")
async def process_audio(
    profile_id: int = Form(...),
    audio_file: UploadFile = File(...),
    separation_model: str = Form("convtasnet"),  # Default to ConvTasNet
    embedding_model: str = Form("resemblyzer"),  # Default to Resemblyzer
    use_vad: bool = Form(False),                 # Default to no VAD
    vad_processor: str = Form("webrtcvad"),      # Default VAD processor
    use_gpu: bool = Form(False),                 # Default to CPU
    save_metrics: bool = Form(False),            # Whether to save performance metrics
    db: Session = Depends(get_db)
):
    """Process audio file to isolate a voice based on profile."""
    logger.info(f"Received audio processing request for profile ID: {profile_id}")
    logger.info(f"Using separation model: {separation_model}")
    logger.info(f"Using embedding model: {embedding_model}")
    logger.info(f"Use VAD: {use_vad}, VAD processor: {vad_processor if use_vad else 'N/A'}")
    logger.info(f"Uploaded file: {audio_file.filename}, content type: {audio_file.content_type}")
    
    # Create temporary file for uploaded audio - preserve original extension
    timestamp = int(time.time())
    file_ext = os.path.splitext(audio_file.filename)[1]
    
    # If no extension provided in filename, try to determine from content type
    if not file_ext:
        if "webm" in audio_file.content_type:
            file_ext = ".webm"
        elif "wav" in audio_file.content_type:
            file_ext = ".wav"
        elif "mp3" in audio_file.content_type:
            file_ext = ".mp3"
        elif "ogg" in audio_file.content_type or "opus" in audio_file.content_type:
            file_ext = ".ogg"
        else:
            file_ext = ".audio"  # Generic extension
    
    temp_orig_path = os.path.join(TEMP_DIR, f"temp_orig_{timestamp}{file_ext}")
    temp_wav_path = os.path.join(TEMP_DIR, f"temp_{timestamp}.wav")
    
    try:
        # Save uploaded file with original format
        logger.info(f"Saving uploaded file to: {temp_orig_path}")
        content = await audio_file.read()
        with open(temp_orig_path, "wb") as buffer:
            buffer.write(content)
        
        # Verify file was saved
        if not os.path.exists(temp_orig_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
            
        file_size = os.path.getsize(temp_orig_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Check if it's already a valid WAV
        is_wav = file_ext.lower() == ".wav" and is_valid_wav(temp_orig_path)
        
        if is_wav:
            # If it's already a valid WAV file, just use the original
            logger.info("File is already a valid WAV - skipping conversion")
            temp_wav_path = temp_orig_path
        else:
            # Convert to WAV format
            logger.info(f"Converting {file_ext} to WAV format")
            try:
                temp_wav_path = convert_audio_to_wav(
                    temp_orig_path, 
                    temp_wav_path,
                    sample_rate=16000  # 16kHz sample rate for voice processing
                )
            except Exception as e:
                logger.error(f"Audio conversion error: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not convert audio file: {str(e)}"
                )
        
        # Save a copy of the mixed audio for history
        mixed_audio_filename = f"mixed_{timestamp}.wav"
        mixed_audio_path = os.path.join(MIXED_DIR, mixed_audio_filename)
        logger.info(f"Saving mixed audio to: {mixed_audio_path}")
        shutil.copy(temp_wav_path, mixed_audio_path)
        
        # Process the audio file to isolate the voice
        logger.info("Starting voice isolation")
        output_dir = "data/isolated"
        
        try:
            # Important: Pass parameters in the correct order
            isolated_path, similarity, metrics = isolate_voice(
                temp_wav_path,
                profile_id,
                db,  # This parameter will be passed as db_session
                output_dir=output_dir,
                use_gpu=use_gpu,
                separation_model_id=separation_model,
                embedding_model_id=embedding_model,
                use_vad=use_vad,
                vad_processor_id=vad_processor,
                return_metrics=save_metrics
            )
            
            # Get the filename for response
            isolated_filename = os.path.basename(isolated_path)
            logger.info(f"Voice isolation complete. Result file: {isolated_filename}")
            
            # Save metrics if requested
            metrics_path = None
            if save_metrics and metrics:
                metrics_filename = f"metrics_{timestamp}.json"
                metrics_path = os.path.join(BENCHMARK_DIR, metrics_filename)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Performance metrics saved to: {metrics_path}")
            
            # Return response with mixed audio path and optional metrics
            response = {
                "status": "success",
                "message": "Audio processed successfully",
                "file_name": isolated_filename,
                "similarity": float(similarity),
                "file_path": f"/data/isolated/{isolated_filename}",
                "mixed_audio_path": f"/data/mixed/{mixed_audio_filename}",
                "separation_model": separation_model,
                "embedding_model": embedding_model,
                "use_vad": use_vad,
                "vad_processor": vad_processor if use_vad else None
            }
            
            # Add metrics path if saved
            if metrics_path:
                response["metrics_path"] = f"/data/benchmarks/{os.path.basename(metrics_path)}"
                response["metrics"] = metrics
            
            return response
            
        except Exception as e:
            logger.error(f"Voice isolation error: {str(e)}")
            # Clean up mixed audio if isolation failed
            if os.path.exists(mixed_audio_path):
                try:
                    os.remove(mixed_audio_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up mixed audio file: {cleanup_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Voice isolation failed: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error for other exceptions
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        
        # Return a helpful error message
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temporary files (but not the mixed audio we want to keep)
        for path in [temp_orig_path, temp_wav_path]:
            if os.path.exists(path) and os.path.basename(path).startswith("temp_"):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")

@router.post("/benchmark")
async def benchmark_models(
    profile_id: int = Form(...),
    audio_file: UploadFile = File(...),
    separation_models: str = Form("convtasnet"),  # Comma-separated list or single model
    embedding_models: str = Form("resemblyzer"),  # Comma-separated list or single model
    use_vad: bool = Form(False),
    vad_processors: str = Form("webrtcvad"),  # Comma-separated list or single processor
    output_format: str = Form("json"),  # json, csv, or markdown
    db: Session = Depends(get_db)
):
    """
    Benchmark multiple model combinations on the same audio file.
    
    This endpoint allows testing different model combinations to compare their performance.
    """
    logger.info(f"Received benchmarking request for profile ID: {profile_id}")
    
    # Parse model lists
    separation_model_list = [m.strip() for m in separation_models.split(",")]
    embedding_model_list = [m.strip() for m in embedding_models.split(",")]
    vad_processor_list = [p.strip() for p in vad_processors.split(",")] if use_vad else ["none"]
    
    logger.info(f"Benchmarking {len(separation_model_list)} separation models: {separation_model_list}")
    logger.info(f"Benchmarking {len(embedding_model_list)} embedding models: {embedding_model_list}")
    if use_vad:
        logger.info(f"Benchmarking {len(vad_processor_list)} VAD processors: {vad_processor_list}")
    
    # Create temporary file for uploaded audio
    timestamp = int(time.time())
    file_ext = os.path.splitext(audio_file.filename)[1]
    
    # If no extension provided, determine from content type
    if not file_ext:
        if "webm" in audio_file.content_type:
            file_ext = ".webm"
        elif "wav" in audio_file.content_type:
            file_ext = ".wav"
        elif "mp3" in audio_file.content_type:
            file_ext = ".mp3"
        elif "ogg" in audio_file.content_type or "opus" in audio_file.content_type:
            file_ext = ".ogg"
        else:
            file_ext = ".audio"
    
    temp_orig_path = os.path.join(TEMP_DIR, f"benchmark_orig_{timestamp}{file_ext}")
    temp_wav_path = os.path.join(TEMP_DIR, f"benchmark_{timestamp}.wav")
    
    try:
        # Save uploaded file
        logger.info(f"Saving uploaded file to: {temp_orig_path}")
        content = await audio_file.read()
        with open(temp_orig_path, "wb") as buffer:
            buffer.write(content)
        
        # Convert to WAV if needed
        is_wav = file_ext.lower() == ".wav" and is_valid_wav(temp_orig_path)
        if is_wav:
            temp_wav_path = temp_orig_path
        else:
            temp_wav_path = convert_audio_to_wav(temp_orig_path, temp_wav_path, sample_rate=16000)
        
        # Create benchmark directory and instantiate benchmark utility
        benchmark_dir = os.path.join(BENCHMARK_DIR, f"benchmark_{timestamp}")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        benchmark_util = ModelBenchmark(benchmark_dir=benchmark_dir)
        
        # Get reference voice profile
        from models.voice_profile import VoiceProfile
        profile = db.query(VoiceProfile).get(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Voice profile with ID {profile_id} not found")
        
        # Save mixed audio
        mixed_audio_filename = f"benchmark_mixed_{timestamp}.wav"
        mixed_audio_path = os.path.join(MIXED_DIR, mixed_audio_filename)
        shutil.copy(temp_wav_path, mixed_audio_path)
        
        # Run benchmarks for all combinations
        results = []
        
        for sep_model in separation_model_list:
            for emb_model in embedding_model_list:
                for vad_proc in vad_processor_list:
                    try:
                        # Skip if VAD processor is "none"
                        use_vad_for_this_run = use_vad and vad_proc != "none"
                        
                        logger.info(f"Benchmarking combination: {sep_model} + {emb_model}" + 
                                   (f" + {vad_proc}" if use_vad_for_this_run else ""))
                        
                        # Process audio with this combination
                        isolated_path, similarity, metrics = isolate_voice(
                            temp_wav_path,
                            profile_id,
                            db,
                            output_dir=benchmark_dir,
                            use_gpu=False,  # Use CPU for consistent benchmarking
                            separation_model_id=sep_model,
                            embedding_model_id=emb_model,
                            use_vad=use_vad_for_this_run,
                            vad_processor_id=vad_proc if use_vad_for_this_run else "webrtcvad",
                            return_metrics=True
                        )
                        
                        # Add to results
                        result = {
                            "separation_model": sep_model,
                            "embedding_model": emb_model,
                            "use_vad": use_vad_for_this_run,
                            "vad_processor": vad_proc if use_vad_for_this_run else None,
                            "similarity": float(similarity),
                            "total_time": metrics["total_time_seconds"],
                            "isolated_path": isolated_path,
                            "metrics": metrics
                        }
                        results.append(result)
                        
                        # Update benchmark results
                        benchmark_util.benchmark_results[
                            f"pipeline_{sep_model}_{emb_model}" + 
                            (f"_{vad_proc}" if use_vad_for_this_run else "")
                        ] = metrics
                        
                    except Exception as e:
                        logger.error(f"Error benchmarking {sep_model}+{emb_model}" + 
                                    (f"+{vad_proc}" if use_vad_for_this_run else "") + f": {e}")
                        results.append({
                            "separation_model": sep_model,
                            "embedding_model": emb_model,
                            "use_vad": use_vad_for_this_run,
                            "vad_processor": vad_proc if use_vad_for_this_run else None,
                            "error": str(e)
                        })
        
        # Save benchmark results
        benchmark_results_path = benchmark_util.save_all_results(f"benchmark_results_{timestamp}.json")
        
        # Generate comparison report
        comparison_path = compare_models(
            benchmark_util.benchmark_results,
            model_type="pipeline",
            output_dir=benchmark_dir,
            output_format=output_format
        )
        
        # Return the results and paths to files
        return {
            "status": "success",
            "message": f"Benchmarked {len(results)} model combinations",
            "results": results,
            "benchmark_results_path": f"/data/benchmarks/{os.path.basename(benchmark_dir)}/{os.path.basename(benchmark_results_path)}",
            "comparison_path": f"/data/benchmarks/{os.path.basename(benchmark_dir)}/{os.path.basename(comparison_path)}",
            "mixed_audio_path": f"/data/mixed/{mixed_audio_filename}"
        }
        
    except Exception as e:
        logger.error(f"Error in benchmarking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")
    finally:
        # Clean up temporary files
        for path in [temp_orig_path, temp_wav_path]:
            if os.path.exists(path) and os.path.basename(path).startswith("benchmark_"):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")