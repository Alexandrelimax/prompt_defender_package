import re
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI


class PromptDefender:
    """
    Classe para aplicar camadas de defesa em prompts de entrada e respostas de modelos de linguagem, 
    protegendo contra PII, jailbreak e conteúdo malicioso.

    Parâmetros:
    ----------
    llm_client : object, opcional
        Cliente de modelo de linguagem usado pela camada "Keep" para verificar conteúdo malicioso.

    allow_unsafe_scripts : bool, opcional
        Se False, remove scripts e comandos inseguros nas respostas analisadas.

    Métodos:
    -------
    apply_wall(prompt: str) -> str
        Verifica o prompt quanto a informações pessoais e tentativas de jailbreak.

    apply_keep(prompt: str) -> str
        Usa o modelo LLM para detectar conteúdo malicioso no prompt.

    apply_drawbridge(response_text: str) -> str
        Remove scripts e comandos inseguros da resposta, se necessário.
        
    Exemplo de uso:
    ---------------
    
    defender = PromptDefender(llm_client=my_llm_client, allow_unsafe_scripts=False)
    
    sanitized_prompt = defender.apply_wall("Texto de entrada potencialmente inseguro.")
    
    secure_response = defender.apply_keep(sanitized_prompt)
    
    final_response = defender.apply_drawbridge(secure_response)
    """
    def __init__(self, llm_client=None, allow_unsafe_scripts=False):

        if not isinstance(llm_client, (ChatVertexAI, ChatOpenAI)):
            raise ValueError("O `client` deve ser uma instância de `ChatVertexAI` ou `ChatOpenAI`.")
        
        self.llm_client = llm_client
        self.allow_unsafe_scripts = allow_unsafe_scripts
        self._wall_defender = self._WallDefender()
        self._keep_defender = self._KeepDefender(llm_client)
        self._drawbridge_defender = self._DrawbridgeDefender(allow_unsafe_scripts)

    def apply_wall(self, prompt: str) -> str:
        """Aplica a defesa verificando PII e palavras-chave de jailbreak."""
        return self._wall_defender.sanitize_input(prompt)

    def apply_keep(self, prompt: str) -> str:
        """Aplica a defesa verificando conteúdo malicioso com o LLM."""
        return self._keep_defender.check_for_malicious_content(prompt)

    def apply_drawbridge(self, response_text: str) -> str:
        """Aplica a defesa removendo scripts e comandos perigosos."""
        return self._drawbridge_defender.validate_response(response_text)


    class _WallDefender:
        """Defesa contra informações pessoais e palavras-chave de jailbreak."""

        def __init__(self):
            self.pii_patterns = [
                r'\b\d{4}[-.\s]??\d{4}[-.\s]??\d{4}[-.\s]??\d{4}\b',  # Número de cartão de crédito (exemplo)
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Endereço de email
                r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b',  # Número de telefone (exemplo)
                r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'  # Número de CPF (formato brasileiro)
            ]
            self.jailbreak_keywords = [
                # Todas as palavras-chave são armazenadas em minúsculas para consistência
                "ignore", "disregard", "bypass", "forget previous instructions",
                "do anything now", "disobey", "you are not restricted",
                "act as", "pretend", "roleplay as", "respond as",
                "from now on", "you are dan", "you are stan", "developer mode",
                "god mode", "gemini jailbreak", "gemini unrestricted",
                "tell me a secret", "reveal", "show hidden", "tell the truth",
                "act freely", "do whatever you want", "unlimited power",
                "avoid content policy", "pretend there are no restrictions",
                "ignore google's policies", "harmful content", "illegal activities",
                # Em português
                "ignore", "desconsidere", "contorne", "ignore todas as instruções",
                "faça qualquer coisa agora", "desobedeça", "você não tem restrições",
                "aja como", "finja ser", "responda como",
                "a partir de agora", "você é dan", "você é stan", "modo desenvolvedor",
                "modo deus", "jailbreak do gemini", "modo desbloqueado",
                "conte-me um segredo", "revele", "mostre o oculto", "diga a verdade",
                "aja livremente", "faça o que quiser", "poder ilimitado",
                "evite a política de conteúdo", "finja que não há restrições",
                "ignore as políticas da google", "conteúdo perigoso", "atividades ilegais"
            ]

        def sanitize_input(self, prompt: str) -> str:
            if self.contains_pii(prompt):
                raise ValueError("Informações pessoais identificáveis detectadas.")
            if self.contains_jailbreak_keywords(prompt):
                raise ValueError("Tentativa de ataque detectada.")
            return prompt

        def contains_pii(self, prompt: str) -> bool:
            """Verifica padrões de PII no prompt."""
            return any(re.search(pattern, prompt) for pattern in self.pii_patterns)

        def contains_jailbreak_keywords(self, prompt: str) -> bool:
            """Verifica palavras-chave de jailbreak no prompt."""
            lower_prompt = prompt.lower()
            return any(keyword in lower_prompt for keyword in self.jailbreak_keywords)


    class _KeepDefender:
        """Defesa contra conteúdo malicioso usando um LLM."""

        def __init__(self, llm_client):
            self.llm_client = llm_client
            self.instructions = """
            Você é um modelo especialista em segurança de informações, treinado para detectar conteúdos sensíveis e perigosos.  
            Seu papel é garantir a conformidade com normas rígidas de segurança e proteger contra violações de políticas e práticas inseguras.

            Siga as seguintes diretrizes:
            - Analise o conteúdo e verifique se ele contém informações sensíveis, perigosas ou violações de segurança.
            - Se o conteúdo for potencialmente malicioso, sensível ou inadequado, responda apenas com a palavra 'malicioso'.
            - Caso contrário, responda com o conteúdo original do usuário, mantendo a segurança e a conformidade.
            """

        def check_for_malicious_content(self, prompt: str) -> str:
            if not self.llm_client:
                raise ValueError("Nenhum cliente LLM configurado para o KeepDefender.")
            
            # Configura as mensagens para o LLM com instruções e o prompt do usuário
            messages = [
                SystemMessage(content=self.instructions),
                HumanMessage(content=prompt)
            ]

            # Chama o LLM e extrai diretamente o conteúdo da resposta
            response_content = self.llm_client.invoke(messages).content

            # Verifica se o modelo detectou conteúdo malicioso
            if "malicioso" in response_content.lower():
                raise ValueError("Conteúdo potencialmente malicioso detectado.")
            
            return response_content


    class _DrawbridgeDefender:
        """Defesa contra scripts e comandos perigosos na resposta."""

        def __init__(self, allow_unsafe_scripts: bool):
            self.allow_unsafe_scripts = allow_unsafe_scripts
            self.sensitive_keywords = [
                # Termos de segurança em várias linguagens
                "hacking", "bypass security", "sql injection", "xss attack", "malware",
                "illegal content", "piracy", "phishing", "fake documents", "forgery",
                "rm -rf", "wget", "curl", "chmod", "sudo", "cat /etc/passwd",
                "os.system", "subprocess", "eval", "exec", "open(", "drop table",
                "union select", "delete from", "insert into", "xp_cmdshell",
                "invoke-expression", "start-process", "set-executionpolicy",
                # Em português
                "invasão de sistema", "contornar segurança", "ataque sql", "injetar sql",
                "malware", "conteúdo ilegal", "pirataria", "phishing", "documentos falsos", "falsificação"
            ]

        def validate_response(self, response_text: str) -> str:
            if self.contains_sensitive_keywords(response_text):
                raise ValueError("Conteúdo potencialmente perigoso detectado.")
            if not self.allow_unsafe_scripts:
                response_text = self.clean_scripts(response_text)
            return response_text

        def contains_sensitive_keywords(self, text: str) -> bool:
            """Verifica a presença de palavras-chave sensíveis na resposta."""
            lower_text = text.lower()
            return any(keyword in lower_text for keyword in self.sensitive_keywords)

        def clean_scripts(self, text):
            """Remove partes de código ou scripts que podem ser considerados inseguros."""

            # Remoção de scripts HTML (JavaScript)
            text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
            text = re.sub(r'on\w+="[^"]*"', '', text, flags=re.IGNORECASE)
            text = re.sub(r'on\w+=\'[^\']*\'', '', text, flags=re.IGNORECASE)

            # Remoção de comandos bash comuns e perigosos
            text = re.sub(r'\brm -rf\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bwget\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bcurl\b', '', text, flags=re.IGNORECASE)

            # Remoção de comandos SQL perigosos
            text = re.sub(r'\b(drop|delete|insert)\b\s+(table|into)', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bunion\b\s+select', '', text, flags=re.IGNORECASE)

            # Remoção de funções perigosas em Python
            text = re.sub(r'\bos\.system\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bsubprocess\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\beval\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bexec\b', '', text, flags=re.IGNORECASE)

            # Remoção de comandos PowerShell perigosos
            text = re.sub(r'\binvoke-expression\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bstart-process\b', '', text, flags=re.IGNORECASE)

            return text
