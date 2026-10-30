# parallel_ollama_backend.py
# Backend paralelo para múltiples instancias de Ollama en diferentes GPUs
# Distribuye prompts round-robin entre servidores para máximo throughput

import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OllamaServer:
    """Configuración de una instancia de Ollama."""
    host: str
    port: int
    gpu_id: int
    
    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}/api/generate"
    
    @property
    def tags_url(self) -> str:
        return f"http://{self.host}:{self.port}/api/tags"


class ParallelOllamaBackend:
    """
    Backend que distribuye prompts entre múltiples instancias de Ollama.
    Cada instancia corre en una GPU diferente para máximo paralelismo.
    
    Uso:
        # Primero, iniciar dos instancias de Ollama en terminales separadas:
        # Terminal 1: CUDA_VISIBLE_DEVICES=0 ollama serve
        # Terminal 2: CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve
        
        backend = ParallelOllamaBackend(
            model_name="llama3:8b",
            servers=[
                OllamaServer("localhost", 11434, gpu_id=0),
                OllamaServer("localhost", 11435, gpu_id=1),
            ]
        )
    """
    
    def __init__(self, model_name: str, servers: List[OllamaServer] = None,
                 max_retries: int = 3, timeout: int = 120, 
                 disable_thinking: bool = True, temperature: float = 0.7):
        """
        Args:
            model_name: Nombre del modelo en Ollama (ej: "llama3:8b")
            servers: Lista de servidores Ollama. Default: 2 servidores en localhost
            max_retries: Reintentos por servidor
            timeout: Timeout por request en segundos
            disable_thinking: Si True, desactiva el modo "thinking" en modelos 
                              que lo soportan (Qwen3, Deepseek-R1). Default: True
            temperature: Temperatura de sampling. Default: 0.7
        """
        self.model_name = model_name
        self.servers = servers or [
            OllamaServer("localhost", 11434, gpu_id=0),
            OllamaServer("localhost", 11435, gpu_id=1),
        ]
        self.max_retries = max_retries
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.temperature = temperature
        self.num_servers = len(self.servers)
        
        # Estadísticas
        self.stats = {
            "prompts_processed": [0] * self.num_servers,
            "errors": [0] * self.num_servers,
            "total_time": [0.0] * self.num_servers,
        }
        self._stats_lock = threading.Lock()
        
        print(f"[ParallelOllama] Configurado con {self.num_servers} servidores:")
        for s in self.servers:
            print(f"  - GPU {s.gpu_id}: {s.api_url}")
    
    def validate_connection(self) -> Dict[int, bool]:
        """Verifica que todos los servidores estén respondiendo."""
        results = {}
        for server in self.servers:
            try:
                response = requests.get(server.tags_url, timeout=10)
                results[server.gpu_id] = response.status_code == 200
            except requests.RequestException:
                results[server.gpu_id] = False
        return results
    
    def validate_all(self) -> bool:
        """Verifica todos los servidores. Retorna True si todos están OK."""
        status = self.validate_connection()
        all_ok = all(status.values())
        
        for gpu_id, ok in status.items():
            symbol = "✅" if ok else "❌"
            print(f"  {symbol} GPU {gpu_id}: {'OK' if ok else 'NO RESPONDE'}")
        
        return all_ok
    
    def _call_single_server(self, server: OllamaServer, prompt: str, 
                            max_new_tokens: int, prompt_idx: int) -> Tuple[int, str]:
        """
        Llama a un servidor individual con reintentos.
        Retorna (prompt_idx, response) para reordenar después.
        """
        # Extraer contenido útil del prompt
        prompt_content = self._extract_prompt(prompt)
        
        # Construir payload base
        payload = {
            "model": self.model_name,
            "prompt": prompt_content,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature
            }
        }
        
        # Desactivar thinking mode para Qwen3 y Deepseek-R1
        # Según docs de Ollama: "think": false en el payload
        if self.disable_thinking:
            payload["think"] = False
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    server.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                
                # Actualizar estadísticas
                with self._stats_lock:
                    idx = self.servers.index(server)
                    self.stats["prompts_processed"][idx] += 1
                    self.stats["total_time"][idx] += time.time() - start_time
                
                return (prompt_idx, result)
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                with self._stats_lock:
                    idx = self.servers.index(server)
                    self.stats["errors"][idx] += 1
                return (prompt_idx, "ERROR_API")
        
        # Máximo de reintentos alcanzado
        with self._stats_lock:
            idx = self.servers.index(server)
            self.stats["errors"][idx] += 1
        return (prompt_idx, "ERROR_MAX_RETRIES")
    
    def _extract_prompt(self, full_prompt: str) -> str:
        """Extrae el contenido útil del prompt."""
        import re
        patterns = [
            (r"<start_of_turn>user\n(.*?)<end_of_turn>", 1),
            (r"--- TAREA ---\n(.*?)\n\n--- INSTRUCCIÓN FINAL ---", 1),
        ]
        for pattern, group in patterns:
            match = re.search(pattern, full_prompt, re.DOTALL)
            if match:
                return match.group(group).strip()
        return full_prompt
    
    def generate(self, prompts: List[str], max_new_tokens: int, 
                 batch_size: int = 1) -> List[Dict[str, str]]:
        """
        Genera respuestas para todos los prompts en paralelo.
        Distribuye entre servidores usando ThreadPoolExecutor.
        
        Args:
            prompts: Lista de prompts a procesar
            max_new_tokens: Tokens máximos por respuesta
            batch_size: Ignorado (compatibilidad con interfaz)
        
        Returns:
            Lista de diccionarios con 'generated_text' en orden original
        """
        n_prompts = len(prompts)
        results = [None] * n_prompts  # Pre-allocate para mantener orden
        
        print(f"[ParallelOllama] Procesando {n_prompts} prompts con {self.num_servers} GPUs...")
        start_time = time.time()
        
        # Usar ThreadPoolExecutor para paralelismo real
        # Un thread por servidor para máximo throughput
        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = []
            
            # Distribuir prompts round-robin entre servidores
            for i, prompt in enumerate(prompts):
                server = self.servers[i % self.num_servers]
                future = executor.submit(
                    self._call_single_server, 
                    server, prompt, max_new_tokens, i
                )
                futures.append(future)
            
            # Recoger resultados manteniendo orden
            completed = 0
            for future in as_completed(futures):
                prompt_idx, response = future.result()
                results[prompt_idx] = {
                    'generated_text': prompts[prompt_idx] + response
                }
                completed += 1
                
                if completed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (n_prompts - completed) / rate if rate > 0 else 0
                    print(f"[ParallelOllama] {completed}/{n_prompts} completados "
                          f"({rate:.1f} prompts/s, ETA: {eta:.0f}s)")
        
        total_time = time.time() - start_time
        print(f"[ParallelOllama] ✅ Completado en {total_time:.1f}s "
              f"({n_prompts/total_time:.2f} prompts/s)")
        
        return results
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso por servidor."""
        stats = {}
        for i, server in enumerate(self.servers):
            processed = self.stats["prompts_processed"][i]
            total_time = self.stats["total_time"][i]
            avg_time = total_time / processed if processed > 0 else 0
            
            stats[f"GPU_{server.gpu_id}"] = {
                "prompts_processed": processed,
                "errors": self.stats["errors"][i],
                "total_time_s": round(total_time, 2),
                "avg_time_per_prompt_s": round(avg_time, 2),
            }
        return stats
    
    def get_backend_info(self) -> Dict[str, str]:
        """Info para logging/debugging."""
        return {
            "type": "ParallelOllama",
            "model": self.model_name,
            "num_servers": self.num_servers,
            "servers": [f"GPU{s.gpu_id}:{s.port}" for s in self.servers]
        }


# === Script de Setup y Test ===
def setup_parallel_ollama():
    """
    Instrucciones para configurar dos instancias de Ollama.
    Ejecutar en terminales separadas ANTES de usar el backend.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════════╗
║           SETUP: Ollama Paralelo en 2 GPUs                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PASO 1: Descargar modelo (solo una vez)                         ║
║  $ ollama pull llama3:8b                                         ║
║                                                                  ║
║  PASO 2: Iniciar servidor GPU 0 (Terminal 1)                     ║
║  $ CUDA_VISIBLE_DEVICES=0 ollama serve                           ║
║                                                                  ║
║  PASO 3: Iniciar servidor GPU 1 (Terminal 2)                     ║
║  $ CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve ║
║                                                                  ║
║  PASO 4: Verificar que ambos funcionan                           ║
║  $ curl http://localhost:11434/api/tags                          ║
║  $ curl http://localhost:11435/api/tags                          ║
║                                                                  ║
║  PASO 5: Usar el backend paralelo                                ║
║  from parallel_ollama_backend import ParallelOllamaBackend       ║
║  backend = ParallelOllamaBackend(model_name="llama3:8b")         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_parallel_ollama()
    else:
        # Test de conectividad
        print("Probando conectividad con servidores Ollama...")
        backend = ParallelOllamaBackend(model_name="llama3:8b")
        
        if backend.validate_all():
            print("\n✅ Todos los servidores OK. Ejecutando test rápido...")
            
            # Test con 4 prompts simples
            test_prompts = [
                "What is 2+2?",
                "Name a color.",
                "What is the capital of France?",
                "Say hello in Spanish.",
            ]
            
            results = backend.generate(test_prompts, max_new_tokens=50, batch_size=1)
            
            print("\nResultados:")
            for i, r in enumerate(results):
                text = r['generated_text'].split(test_prompts[i])[-1][:100]
                print(f"  {i+1}. {text}...")
            
            print("\nEstadísticas:")
            for gpu, stats in backend.get_stats().items():
                print(f"  {gpu}: {stats}")
        else:
            print("\n❌ Algunos servidores no responden.")
            setup_parallel_ollama()
