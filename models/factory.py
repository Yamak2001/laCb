# models/factory.py
from typing import Dict, Type, Any, Optional
import logging

# Import base classes
from models.base import BaseSeparationModel, BaseEmbeddingModel, BaseVADProcessor

# Initialize logger
logger = logging.getLogger(__name__)

class ModelFactory:
    """Base factory class for model creation"""
    
    @classmethod
    def register_models(cls) -> None:
        """Register available models - implemented by subclasses"""
        pass


class SeparationModelFactory(ModelFactory):
    """Factory class for creating audio source separation models"""
    
    # Dictionary to store registered separation model classes
    _models: Dict[str, Type[BaseSeparationModel]] = {}
    
    @classmethod
    def register(cls, model_id: str, model_class: Type[BaseSeparationModel]) -> None:
        """
        Register a separation model class
        
        Args:
            model_id (str): Unique identifier for the model
            model_class (Type[BaseSeparationModel]): Model class
        """
        cls._models[model_id] = model_class
        logger.info(f"Registered separation model: {model_id}")
    
    @classmethod
    def get_model(cls, model_id: str, **kwargs) -> BaseSeparationModel:
        """
        Create a separation model instance
        
        Args:
            model_id (str): Identifier of the registered model
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            BaseSeparationModel: An instance of the requested model
            
        Raises:
            ValueError: If the model_id is not registered
        """
        if model_id not in cls._models:
            available_models = list(cls._models.keys())
            logger.error(f"Model '{model_id}' not registered. Available models: {available_models}")
            raise ValueError(f"Separation model '{model_id}' not registered")
        
        logger.info(f"Creating separation model: {model_id}")
        return cls._models[model_id](model_name=model_id, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        List all registered models
        
        Returns:
            Dict[str, str]: Dictionary mapping model_id to model class name
        """
        return {model_id: model_class.__name__ for model_id, model_class in cls._models.items()}
    
    @classmethod
    def register_models(cls) -> None:
        """Register all available separation models"""
        try:
            # Import all model implementations here
            from models.separation.convtasnet_model import ConvTasNetModel
            cls.register("convtasnet", ConvTasNetModel)
            
            
            # Import SepFormer if available
            try:
                from models.separation.sepformer_model import SepFormerModel
                cls.register("sepformer", SepFormerModel)
            except ImportError:
                logger.warning("SepFormer model not available. Skip registration.")
            
            # Import DPRNN if available
            try:
                from models.separation.dprnn_model import DPRNNModel
                cls.register("dprnn", DPRNNModel)
            except ImportError:
                logger.warning("DPRNN model not available. Skip registration.")
            
            # Import NeMo models if available
            try:
                from models.separation.nemo_model import NeMoSeparationModel
                cls.register("nemo", NeMoSeparationModel)
            except ImportError:
                logger.warning("NeMo separation model not available. Skip registration.")
            
                
        except Exception as e:
            logger.error(f"Error registering separation models: {e}")


class EmbeddingModelFactory(ModelFactory):
    """Factory class for creating voice embedding models"""
    
    # Dictionary to store registered embedding model classes
    _models: Dict[str, Type[BaseEmbeddingModel]] = {}
    
    @classmethod
    def register(cls, model_id: str, model_class: Type[BaseEmbeddingModel]) -> None:
        """
        Register an embedding model class
        
        Args:
            model_id (str): Unique identifier for the model
            model_class (Type[BaseEmbeddingModel]): Model class
        """
        cls._models[model_id] = model_class
        logger.info(f"Registered embedding model: {model_id}")
    
    @classmethod
    def get_model(cls, model_id: str, **kwargs) -> BaseEmbeddingModel:
        """
        Create an embedding model instance
        
        Args:
            model_id (str): Identifier of the registered model
            **kwargs: Additional parameters to pass to the model constructor
            
        Returns:
            BaseEmbeddingModel: An instance of the requested model
            
        Raises:
            ValueError: If the model_id is not registered
        """
        if model_id not in cls._models:
            available_models = list(cls._models.keys())
            logger.error(f"Model '{model_id}' not registered. Available models: {available_models}")
            raise ValueError(f"Embedding model '{model_id}' not registered")
        
        logger.info(f"Creating embedding model: {model_id}")
        return cls._models[model_id](model_name=model_id, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        List all registered models
        
        Returns:
            Dict[str, str]: Dictionary mapping model_id to model class name
        """
        return {model_id: model_class.__name__ for model_id, model_class in cls._models.items()}
    
    @classmethod
    def register_models(cls) -> None:
        """Register all available embedding models"""
        try:
            # Import all model implementations here
            from models.embedding.resemblyzer_model import ResemblyzerModel
            cls.register("resemblyzer", ResemblyzerModel)
            
            # Import ECAPA-TDNN if available
            try:
                from models.embedding.ecapa_model import ECAPAModel
                cls.register("ecapa", ECAPAModel)
            except ImportError:
                logger.warning("ECAPA-TDNN model not available. Skip registration.")
            
            # Import Pyannote.audio if available
            try:
                from models.embedding.pyannote_model import PyannoteSpeakerModel
                cls.register("pyannote", PyannoteSpeakerModel)
            except ImportError:
                logger.warning("Pyannote embedding model not available. Skip registration.")
            
            # Import SpeechBrain if available
            try:
                from models.embedding.speechbrain_model import SpeechBrainModel
                cls.register("speechbrain", SpeechBrainModel)
            except ImportError:
                logger.warning("SpeechBrain embedding model not available. Skip registration.")
            
            # Import TitaNet if available
            try:
                from models.embedding.titanet_model import TitaNetModel
                cls.register("titanet", TitaNetModel)
            except ImportError:
                logger.warning("TitaNet model not available. Skip registration.")
                
        except Exception as e:
            logger.error(f"Error registering embedding models: {e}")


class VADProcessorFactory(ModelFactory):
    """Factory class for creating Voice Activity Detection processors"""
    
    # Dictionary to store registered VAD processor classes
    _processors: Dict[str, Type[BaseVADProcessor]] = {}
    
    @classmethod
    def register(cls, processor_id: str, processor_class: Type[BaseVADProcessor]) -> None:
        """
        Register a VAD processor class
        
        Args:
            processor_id (str): Unique identifier for the processor
            processor_class (Type[BaseVADProcessor]): Processor class
        """
        cls._processors[processor_id] = processor_class
        logger.info(f"Registered VAD processor: {processor_id}")
    
    @classmethod
    def get_processor(cls, processor_id: str, **kwargs) -> BaseVADProcessor:
        """
        Create a VAD processor instance
        
        Args:
            processor_id (str): Identifier of the registered processor
            **kwargs: Additional parameters to pass to the processor constructor
            
        Returns:
            BaseVADProcessor: An instance of the requested processor
            
        Raises:
            ValueError: If the processor_id is not registered
        """
        if processor_id not in cls._processors:
            available_processors = list(cls._processors.keys())
            logger.error(f"Processor '{processor_id}' not registered. Available processors: {available_processors}")
            raise ValueError(f"VAD processor '{processor_id}' not registered")
        
        logger.info(f"Creating VAD processor: {processor_id}")
        return cls._processors[processor_id](model_name=processor_id, **kwargs)
    
    @classmethod
    def list_available_processors(cls) -> Dict[str, str]:
        """
        List all registered processors
        
        Returns:
            Dict[str, str]: Dictionary mapping processor_id to processor class name
        """
        return {processor_id: processor_class.__name__ for processor_id, processor_class in cls._processors.items()}
    
    @classmethod
    def register_models(cls) -> None:
        """Register all available VAD processors"""
        try:
            # Import all processor implementations here
            from models.vad.webrtcvad_processor import WebRtcVADProcessor
            cls.register("webrtcvad", WebRtcVADProcessor)
            
            # Import Pyannote VAD if available
            try:
                from models.vad.pyannote_vad_processor import PyannoteSpeechDetectionProcessor
                cls.register("pyannote_vad", PyannoteSpeechDetectionProcessor)
            except ImportError:
                logger.warning("Pyannote VAD processor not available. Skip registration.")
                
            # Import SpeechBrain VAD if available
            try:
                from models.vad.speechbrain_vad_processor import SpeechBrainVADProcessor
                cls.register("speechbrain_vad", SpeechBrainVADProcessor)
            except ImportError:
                logger.warning("SpeechBrain VAD processor not available. Skip registration.")
                
        except Exception as e:
            logger.error(f"Error registering VAD processors: {e}")


def register_all_models():
    """Register all available models across all factories"""
    SeparationModelFactory.register_models()
    EmbeddingModelFactory.register_models()
    VADProcessorFactory.register_models()