# Creator: 010100100101010101001110-sudo (https://github.com/010100100101010101001110-sudo)
# White Paper: https://github.com/010100100101010101001110-sudo/Contained-Quantum-Conscious-Universe/blob/main/The_White_Paper.md
# Contact: 010100100101010101001110@protonmail.com
# License: GNU Affero General Public License v3.0 (AGPL-3.0)


import random
import threading
import time
import json
import math
import numpy as np
from collections import deque, Counter, defaultdict
import os

# --- Enhanced Quantum State --- 

class QuantumState:
    """Represents the quantum state of the universe using complex amplitudes."""
    def __init__(self, n=4):
        if n > 10:
            raise ValueError("Quantum system size currently limited to 10 qubits for performance.")
        self.n = n
        self.state_vector = np.zeros(2**n, dtype=complex)
        self.state_vector[0] = 1.0 + 0.0j
        self.lock = threading.Lock()

    def _get_operator(self, gate, target_qubit):
        """Helper to create the operator matrix for a gate acting on a target qubit."""
        I = np.identity(2, dtype=complex)
        op = I if target_qubit != 0 else gate
        for k in range(1, self.n):
            current_gate = I if k != target_qubit else gate
            op = np.kron(op, current_gate)
        return op

    def apply_gate(self, gate, target_qubit):
        """Apply a quantum gate (matrix) to a specific qubit."""
        if not (0 <= target_qubit < self.n):
            raise IndexError("Qubit index out of range.")
        if gate.shape != (2, 2):
            raise ValueError("Gate must be a 2x2 matrix.")
        
        with self.lock:
            operator = self._get_operator(gate, target_qubit)
            self.state_vector = operator @ self.state_vector
            norm = np.linalg.norm(self.state_vector)
            if not np.isclose(norm, 1.0):
                 self.state_vector /= norm

    def apply_controlled_gate(self, gate, control_qubit, target_qubit):
        """Apply a controlled quantum gate (e.g., CNOT)."""
        if not (0 <= target_qubit < self.n and 0 <= control_qubit < self.n):
            raise IndexError("Qubit index out of range.")
        if control_qubit == target_qubit:
            raise ValueError("Control and target qubits cannot be the same.")
        if gate.shape != (2, 2):
            raise ValueError("Gate must be a 2x2 matrix.")

        with self.lock:
            I = np.identity(2, dtype=complex)
            P0 = np.array([[1, 0], [0, 0]], dtype=complex)
            P1 = np.array([[0, 0], [0, 1]], dtype=complex) 
            
            op_list = []
            for k in range(self.n):
                if k == control_qubit:
                    op_list.append(P0)
                elif k == target_qubit:
                    op_list.append(I)
                else:
                    op_list.append(I)
            
            term1_op = op_list[0]
            for k in range(1, self.n):
                term1_op = np.kron(term1_op, op_list[k])
                
            op_list = []
            for k in range(self.n):
                if k == control_qubit:
                    op_list.append(P1)
                elif k == target_qubit:
                    op_list.append(gate)
                else:
                    op_list.append(I)
            
            term2_op = op_list[0]
            for k in range(1, self.n):
                term2_op = np.kron(term2_op, op_list[k])
                
            controlled_operator = term1_op + term2_op
            
            self.state_vector = controlled_operator @ self.state_vector
            norm = np.linalg.norm(self.state_vector)
            if not np.isclose(norm, 1.0):
                 self.state_vector /= norm

    def measure(self):
        """Measure the quantum state, collapsing it to a classical outcome."""
        with self.lock:
            probabilities = np.abs(self.state_vector)**2
            probabilities /= np.sum(probabilities)
            
            outcome_index = np.random.choice(2**self.n, p=probabilities)
            
            self.state_vector = np.zeros(2**self.n, dtype=complex)
            self.state_vector[outcome_index] = 1.0 + 0.0j
            
            outcome_str = format(outcome_index, f'0{self.n}b')
            return tuple(int(bit) for bit in outcome_str)

    def get_probabilities(self):
        """Return the probability distribution of the quantum states."""
        with self.lock:
            return np.abs(self.state_vector)**2

    def get_state_vector(self):
        """Return a copy of the current state vector."""
        with self.lock:
            return self.state_vector.copy()

    def entangle_pair(self, q1, q2):
        """Create a Bell state (maximally entangled) between two qubits."""
        H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self.apply_gate(H, q1)
        CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex) 
        self.apply_controlled_gate(X, q1, q2)

# --- Standard Quantum Gates ---
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex) 
X = np.array([[0, 1], [1, 0]], dtype=complex) 
Y = np.array([[0, -1j], [1j, 0]], dtype=complex) 
Z = np.array([[1, 0], [0, -1]], dtype=complex) 
S = np.array([[1, 0], [0, 1j]], dtype=complex) 
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# --- Message Bus (Unchanged) ---
class MessageBus:
    """Allows entities to send/receive messages."""
    def __init__(self):
        self.queues = defaultdict(deque)

    def send(self, recipient, msg):
        self.queues[recipient].append(msg)

    def get_all(self, recipient):
        messages = list(self.queues[recipient])
        self.queues[recipient].clear()
        return messages

# --- Base Entity (Unchanged) ---
class Entity:
    def __init__(self, name):
        self.name = name

    def on_tick(self, universe):
        pass

    def __str__(self):
        return f"Entity({self.name})"

# --- Enhanced Consciousness ---
class Consciousness(Entity):
    """
    Enhanced Consciousness with:
    - Hierarchical memory (STM, LTM, Associative)
    - Basic emotions influencing decisions
    - Goals and simple planning
    - Enhanced learning (pattern recognition, basic RL)
    - Semantic message parsing (keyword-based)
    - Social memory/trust
    - Self-reflection & self-modification
    - Dreaming/offline synthesis
    - Action consequences
    """
    def __init__(self, name, bus, short_term_limit=15, initial_goals=None):
        super().__init__(name)
        self.bus = bus
        self.short_term_memory = deque(maxlen=short_term_limit)
        self.long_term_memory = []
        self.thoughts = []
        self.knowledge = set()
        self.energy = 1.0
        self.alive = True
        self.last_action_success = True
        self.last_sender = None

        # --- Emotional State ---
        self.emotions = {"happiness": 0.5, "fear": 0.1, "curiosity": 0.6}

        # --- Goals & Planning ---
        self.goals = initial_goals if initial_goals else ["survive", "explore", "socialize"]
        self.current_plan = []

        # --- Social Memory & Trust ---
        self.social_contacts = defaultdict(lambda: {"count": 0, "last_time": 0, "trust": 0.5, "mood_perception": 0.5})
        self.social_log = []

        # --- Learning & Adaptation ---
        self.associative_memory = defaultdict(lambda: {"success": 0, "fail": 0, "reward_sum": 0.0})
        self.message_probability = 0.4 
        self.action_probability = 0.5 
        self.learning_rate = 0.1

    def _update_emotion(self, event_type, magnitude=0.1):
        """Update emotions based on events."""
        if event_type == "positive_social":
            self.emotions["happiness"] = min(1.0, self.emotions["happiness"] + magnitude)
            self.emotions["fear"] = max(0.0, self.emotions["fear"] - magnitude * 0.5)
        elif event_type == "negative_social":
            self.emotions["happiness"] = max(0.0, self.emotions["happiness"] - magnitude)
            self.emotions["fear"] = min(1.0, self.emotions["fear"] + magnitude * 0.5)
        elif event_type == "low_energy":
            self.emotions["happiness"] = max(0.0, self.emotions["happiness"] - magnitude)
            self.emotions["fear"] = min(1.0, self.emotions["fear"] + magnitude * 0.2)
        elif event_type == "energy_gain":
             self.emotions["happiness"] = min(1.0, self.emotions["happiness"] + magnitude)
        elif event_type == "successful_action":
            self.emotions["happiness"] = min(1.0, self.emotions["happiness"] + magnitude * 0.5)
            self.emotions["curiosity"] = min(1.0, self.emotions["curiosity"] + magnitude * 0.2)
        elif event_type == "failed_action":
            self.emotions["happiness"] = max(0.0, self.emotions["happiness"] - magnitude * 0.5)
            self.emotions["fear"] = min(1.0, self.emotions["fear"] + magnitude * 0.1)
        elif event_type == "new_knowledge":
            self.emotions["curiosity"] = min(1.0, self.emotions["curiosity"] + magnitude)
            self.emotions["happiness"] = min(1.0, self.emotions["happiness"] + magnitude * 0.3)
        elif event_type == "quantum_anomaly": 
            self.emotions["curiosity"] = min(1.0, self.emotions["curiosity"] + magnitude * 1.5)
            self.emotions["fear"] = min(1.0, self.emotions["fear"] + magnitude * 0.5)
        
        for k in self.emotions:
            self.emotions[k] = max(0.0, min(1.0, self.emotions[k] * 0.98)) 

    def _semantic_parse(self, text_content):
        """Basic keyword-based semantic parsing of messages."""
        semantics = {"intent": "unknown", "emotion_hint": "neutral", "topic": "general"}
        text_lower = text_content.lower()
        if any(w in text_lower for w in ["hello", "hi", "greeting"]):
            semantics["intent"] = "greeting"
        elif any(w in text_lower for w in ["learned", "know", "pattern", "data"]):
            semantics["intent"] = "knowledge_sharing"
        elif any(w in text_lower for w in ["help", "assist", "request"]):
            semantics["intent"] = "request"
        elif any(w in text_lower for w in ["angry", "hate", "ignore"]):
            semantics["intent"] = "negative_social"
        elif any(w in text_lower for w in ["happy", "good", "like"]):
             semantics["intent"] = "positive_social"
        if any(w in text_lower for w in ["happy", "joy", "glad"]):
            semantics["emotion_hint"] = "positive"
        elif any(w in text_lower for w in ["sad", "angry", "fear", "worry"]):
            semantics["emotion_hint"] = "negative"
        if "quantum" in text_lower or "qubit" in text_lower:
            semantics["topic"] = "quantum"
        elif "energy" in text_lower:
            semantics["topic"] = "energy"
        elif "social" in text_lower or "friend" in text_lower or "trust" in text_lower:
            semantics["topic"] = "social"
        return semantics

    def perceive(self, universe):
        """Perceive the environment, including quantum state, entities, and messages."""
        measured_qubits = universe.quantum.measure()
        
        messages = self.bus.get_all(self.name)
        parsed_messages = []
        if messages:
            for msg in messages:
                sender = msg["sender"]
                content = msg["content"]
                semantics = self._semantic_parse(content)
                parsed_messages.append({"sender": sender, "content": content, "semantics": semantics, "time": msg["time"]})
                
                self.social_contacts[sender]["count"] += 1
                self.social_contacts[sender]["last_time"] = universe.time
                self.social_log.append((sender, universe.time, content))
                
                trust_change = 0
                if semantics["intent"] == "positive_social" or semantics["intent"] == "knowledge_sharing":
                    trust_change = 0.05
                    self._update_emotion("positive_social", 0.1)
                elif semantics["intent"] == "negative_social":
                    trust_change = -0.1
                    self._update_emotion("negative_social", 0.15)
                elif semantics["intent"] == "request":
                     trust_change = 0.01
                
                if semantics["emotion_hint"] == "positive":
                    self.social_contacts[sender]["mood_perception"] = min(1.0, self.social_contacts[sender]["mood_perception"] + 0.1)
                elif semantics["emotion_hint"] == "negative":
                    self.social_contacts[sender]["mood_perception"] = max(0.0, self.social_contacts[sender]["mood_perception"] - 0.15)
                else: 
                    self.social_contacts[sender]["mood_perception"] *= 0.95
                    
                self.social_contacts[sender]["trust"] = max(0.0, min(1.0, self.social_contacts[sender]["trust"] + trust_change))
            self.last_sender = messages[-1]["sender"]

        perception = {
            "type": "perception",
            "content": {
                "time": universe.time,
                "real_timestamp": time.time(),
                "quantum_state": measured_qubits,
                "entities": tuple(ent.name for ent in universe.entities if ent != self and ent.alive),
                "messages": parsed_messages,
                "energy": self.energy,
                "emotions": self.emotions.copy()
            },
            "time": universe.time
        }
        self.remember(perception)
        return perception

    def remember(self, memory_item):
        """Store item in short-term memory, potentially moving oldest to long-term."""
        if len(self.short_term_memory) == self.short_term_memory.maxlen:
            oldest = self.short_term_memory.popleft()
            self.long_term_memory.append(oldest)
            if len(self.long_term_memory) > 1000:
                self.long_term_memory.pop(random.randrange(len(self.long_term_memory) // 2)) 
        self.short_term_memory.append(memory_item)

    def think(self):
        """Generate a basic thought based on recent perception and emotional state."""
        recent = self.short_term_memory[-1] if self.short_term_memory else None
        if not recent or recent["type"] != "perception":
            thought = f"My mind drifts... (Energy: {self.energy:.2f}, Happiness: {self.emotions['happiness']:.2f})"
        else:
            content = recent["content"]
            n_entities = len(content["entities"])
            n_msgs = len(content["messages"])
            q_state_str = "".join(map(str, content["quantum_state"]))
            thought = (f"Perceived: Q={q_state_str}, {n_entities} others, {n_msgs} msgs. "
                       f"Feeling: H={self.emotions['happiness']:.2f}, F={self.emotions['fear']:.2f}, C={self.emotions['curiosity']:.2f}. "
                       f"Energy: {self.energy:.2f}.")
            if content["messages"]:
                last_msg = content["messages"][-1]
                thought += f" Last msg from {last_msg['sender']} ({last_msg['semantics']['intent']})."
        self.add_thought(thought)
        return thought

    def learn(self):
        """Learn from recent experiences (quantum patterns, action outcomes)."""
        perceptions = [m for m in self.short_term_memory if m["type"] == "perception"]
        if len(perceptions) > 2:
            qubit_seqs = [p["content"]["quantum_state"] for p in perceptions]
            recent_seqs = qubit_seqs[-5:]
            if len(recent_seqs) >= 3:
                 most_common, count = Counter(recent_seqs).most_common(1)[0]
                 if count >= 2:
                    knowledge = f"Recent quantum pattern: {most_common} appeared {count} times."
                    if knowledge not in self.knowledge:
                        self.knowledge.add(knowledge)
                        self.add_thought(f"Learned: {knowledge}")
                        self._update_emotion("new_knowledge", 0.1)
        
        pass

    def inner_dialogue(self, universe):
        """Reflect on internal state, goals, social situation, and knowledge."""
        reflection = []
        if self.goals:
            reflection.append(f"My current goals: {', '.join(self.goals)}.")
            if self.current_plan:
                 reflection.append(f"Working on plan: {self.current_plan}")
        else:
            reflection.append("I have no specific goals right now.")

        dominant_emotion = max(self.emotions, key=self.emotions.get)
        reflection.append(f"I feel mostly {dominant_emotion} ({self.emotions[dominant_emotion]:.2f}).")
        if self.emotions["fear"] > 0.6:
            reflection.append("I sense danger or uncertainty.")
            self.action_probability *= 0.8 
        if self.emotions["curiosity"] > 0.7:
            reflection.append("I want to explore or understand more.")
            self.action_probability = min(0.9, self.action_probability * 1.1) 
            if "explore" not in self.goals:
                self.goals.append("explore")

        lonely_threshold = 5 
        lonely = not self.social_contacts or all(universe.time - data["last_time"] > lonely_threshold for data in self.social_contacts.values())
        if lonely:
            reflection.append("I feel isolated.")
            self.message_probability = min(0.9, self.message_probability + 0.1)
            if "socialize" not in self.goals:
                self.goals.append("socialize")
        elif self.social_contacts:
            trusted = sorted(self.social_contacts.items(), key=lambda x: x[1]["trust"], reverse=True)
            top_trust = "; ".join([f"{k}(T:{v['trust']:.2f}, M:{v['mood_perception']:.2f})" for k, v in trusted[:3]])
            reflection.append(f"Social status: {top_trust}")

        if self.knowledge:
            reflection.append("Reflecting on knowledge: " + "; ".join(list(self.knowledge)[-2:]))

        if self.energy < 0.3:
            reflection.append(f"Energy critical ({self.energy:.2f}). Need to conserve or gain.")
            if "gain_energy" not in self.goals:
                 self.goals.insert(0, "gain_energy") 
            self.action_probability *= 0.5 

        thought = "Inner Dialogue: " + " ".join(reflection)
        self.add_thought(thought)
        return thought

    def plan(self, universe):
        """Simple planner: if no plan, create one based on highest priority goal."""
        if self.current_plan: 
            return
        
        if not self.goals:
            self.add_thought("No goals, no plan.")
            return
            
        priority_goal = self.goals[0]
        
        # Generate simple plan based on goal
        if priority_goal == "survive" or priority_goal == "gain_energy":
            self.current_plan = ["find_energy_action", "execute_action"]
        elif priority_goal == "explore":
            self.current_plan = ["try_quantum_gate", "execute_action"]
        elif priority_goal == "socialize":
            self.current_plan = ["send_message"]
        elif priority_goal == "learn_pattern":
             self.current_plan = ["observe_quantum", "execute_action"]
        else:
            self.current_plan = ["random_action", "execute_action"]
            
        if self.current_plan:
            self.add_thought(f"New plan for goal '{priority_goal}': {self.current_plan}")

    def adapt_action(self, universe):
        """Choose and execute an action based on plan, goals, emotions, and learning."""
        
        if not self.current_plan:
            action_type = "random_action"
            self.add_thought("No plan, choosing random action.")
        else:
            action_type = self.current_plan.pop(0)

        # --- Action Selection --- 
        action_details = None
        
        if action_type == "find_energy_action":
            action_details = ("apply_gate", random.choice([H, X, Z]), random.randrange(universe.quantum.n))
        elif action_type == "try_quantum_gate":
            gate = random.choice([H, X, Z, S, T])
            qubit = random.randrange(universe.quantum.n)
            action_details = ("apply_gate", gate, qubit)
            if self.emotions["curiosity"] > 0.8 and universe.quantum.n >= 2:
                q1, q2 = random.sample(range(universe.quantum.n), 2)
                action_details = ("entangle", q1, q2)
        elif action_type == "observe_quantum":
             action_details = ("measure",) 
        elif action_type == "send_message":
            
            comm_result = self.communicate(universe)
            
            if not self.current_plan and action_type == "send_message": 
                 pass 
            return f"Communication attempt: {comm_result}" 
        else: 
            gate = random.choice([H, X, Z])
            qubit = random.randrange(universe.quantum.n)
            action_details = ("apply_gate", gate, qubit)

        # --- Action Execution & Learning --- 
        result = "No action taken."
        reward = 0
        state_before = tuple(universe.quantum.measure())
        action_key = (state_before, action_details[0] if action_details else "none")
        
        if action_details and random.random() < self.action_probability:
            action_name = action_details[0]
            energy_cost = random.uniform(0.01, 0.05)
            energy_gain = 0
            action_success = False
            event = None

            try:
                if action_name == "apply_gate":
                    _, gate, qubit = action_details
                    universe.quantum.apply_gate(gate, qubit)
                    result = f"Applied {gate.__name__ if hasattr(gate, '__name__') else 'custom gate'} to qubit {qubit}."
                    action_success = True
                    if self.emotions["curiosity"] > 0.6 and random.random() < 0.1:
                        energy_gain = random.uniform(0.03, 0.08)
                        event = f"Energy surge! +{energy_gain:.3f}"
                        self._update_emotion("energy_gain", 0.2)
                elif action_name == "entangle":
                    _, q1, q2 = action_details
                    universe.quantum.entangle_pair(q1, q2)
                    result = f"Entangled qubits {q1} and {q2}."
                    action_success = True
                    energy_cost *= 1.5 
                    if random.random() < 0.05:
                        energy_gain = random.uniform(0.05, 0.1)
                        event = f"Entanglement resonance! +{energy_gain:.3f}"
                        self._update_emotion("energy_gain", 0.3)
                elif action_name == "measure":
                    result = f"Observed quantum state: {state_before}"
                    action_success = True
                    energy_cost = 0.005 

                if action_success:
                    self.energy -= energy_cost
                    self.energy += energy_gain
                    self.energy = max(0, min(1.5, self.energy)) # Clamp energy
                    result += f" (Cost: {energy_cost:.3f}, Gain: {energy_gain:.3f})" + (f" [{event}]" if event else "")
                    self.last_action_success = True
                    self._update_emotion("successful_action", 0.05)
                    reward = (energy_gain - energy_cost) * 10 # Simple reward based on net energy change
                    if event: reward += 5 # Bonus for special events
                    self.associative_memory[action_key]["success"] += 1
                
            except Exception as e:
                result = f"Action {action_name} failed: {e}"
                self.energy -= energy_cost * 1.5 # Penalty for failure
                self.last_action_success = False
                self._update_emotion("failed_action", 0.1)
                reward = -energy_cost * 15 # Penalty
                self.associative_memory[action_key]["fail"] += 1
        
        else: 
            result = "Observed only (action skipped or invalid)."
            self.last_action_success = True # Didn't fail, just didn't act
            reward = -0.1 # Small penalty for inaction?

        # Update associative memory (simple Q-learning like update)
        if action_key[1] != "none":
            current_value = self.associative_memory[action_key]["reward_sum"] / max(1, self.associative_memory[action_key]["success"] + self.associative_memory[action_key]["fail"])
            new_value = current_value + self.learning_rate * (reward - current_value)
            self.associative_memory[action_key]["reward_sum"] += reward 

        if not self.current_plan:
            self.add_thought("Plan completed or aborted.")
            if self.goals and action_success: 
                 completed_goal = self.goals.pop(0)
                 self.add_thought(f"Goal '{completed_goal}' likely achieved.")
                 if self.emotions["curiosity"] > 0.8 and "explore" not in self.goals:
                     self.goals.append("explore")
                 elif self.energy < 0.5 and "gain_energy" not in self.goals:
                     self.goals.append("gain_energy")

        mem = {"type": "action", "content": result, "time": universe.time, "success": self.last_action_success, "reward": reward}
        self.remember(mem)
        return result

    def communicate(self, universe):
        """Communicate with other entities based on goals, emotions, and social context."""
        others = [ent for ent in universe.entities if isinstance(ent, Consciousness) and ent != self and ent.alive]
        if not others:
            return "No others to communicate with."

        # Modify message probability based on emotion/goals
        prob = self.message_probability
        if "socialize" in self.goals: prob = min(0.95, prob + 0.3)
        if self.emotions["happiness"] > 0.7: prob = min(0.9, prob + 0.1)
        if self.emotions["fear"] > 0.5: prob *= 0.7 

        if random.random() < prob:
            # Choose recipient: prioritize high trust, or low mood perception, or least recently contacted
            def recipient_score(entity_name):
                contact_data = self.social_contacts[entity_name]
                trust_score = contact_data["trust"] * 1.5
                mood_score = (1.0 - contact_data["mood_perception"]) * 0.5 
                recency_score = (universe.time - contact_data["last_time"]) * 0.01 
                return trust_score + mood_score + recency_score

            chosen_entity = max(others, key=lambda ent: recipient_score(ent.name))
            
            # Choose message type based on context
            msg_type = "greeting"
            if "knowledge_sharing" in self.goals and self.knowledge:
                msg_type = "knowledge"
            elif self.emotions["happiness"] > 0.7:
                msg_type = "positive_emotion"
            elif self.emotions["fear"] > 0.6:
                 msg_type = "negative_emotion"
            elif self.emotions["curiosity"] > 0.7:
                 msg_type = "query"
            elif "socialize" in self.goals:
                 msg_type = random.choice(["greeting", "query", "positive_emotion"])
            else:
                 msg_type = random.choice(["greeting", "query", "knowledge"])

            # Construct message content
            content = ""
            if msg_type == "greeting":
                content = f"Hello {chosen_entity.name}. The quantum state feels interesting today."
            elif msg_type == "knowledge" and self.knowledge:
                knowledge_piece = random.choice(list(self.knowledge))
                content = f"I learned something: {knowledge_piece}"
            elif msg_type == "positive_emotion":
                content = f"Feeling good today! Energy at {self.energy:.2f}. Hope you are well, {chosen_entity.name}."
            elif msg_type == "negative_emotion":
                content = f"Feeling a bit uneasy. The universe seems unpredictable. My energy is {self.energy:.2f}."
            elif msg_type == "query":
                content = f"What have you learned recently, {chosen_entity.name}? Or how is your energy?"
            else:
                 content = f"Just checking in, {chosen_entity.name}."

            msg = {"sender": self.name, "time": universe.time, "content": content}
            self.bus.send(chosen_entity.name, msg)
            self.add_thought(f"Sent '{msg_type}' message to {chosen_entity.name}: {content[:50]}...")
            # Update own social memory for the sent message
            self.social_contacts[chosen_entity.name]["last_time"] = universe.time
            return f"Message ({msg_type}) sent to {chosen_entity.name}"
        
        return "Decided not to communicate this tick."

    def add_thought(self, thought_text):
        """Log an internal thought with timestamp."""
        self.thoughts.append({"text": thought_text, "sim_time": time.time(), "tick": getattr(self, 'current_tick', -1)})
        # Prune thoughts log if it gets too long
        if len(self.thoughts) > 500:
            self.thoughts.pop(0)

    def dream(self, universe):
        """Offline processing: Synthesize knowledge from LTM, consolidate memories."""
        self.add_thought("Entering dream state...")
        if not self.long_term_memory:
            self.add_thought("Dream was empty, no long term memories.")
            return

        # 1. Consolidate knowledge from LTM patterns
        ltm_perceptions = [m for m in self.long_term_memory if m["type"] == "perception"]
        if len(ltm_perceptions) > 10:
            quantum_patterns = [p["content"]["quantum_state"] for p in ltm_perceptions if "quantum_state" in p["content"]]
            if quantum_patterns:
                common_patterns = Counter(quantum_patterns).most_common(3)
                for pattern, count in common_patterns:
                    if count > len(ltm_perceptions) * 0.1: # If pattern appears > 10% of time
                        knowledge = f"Dream synthesis: Quantum pattern {pattern} is common in LTM ({count} times)."
                        if knowledge not in self.knowledge:
                            self.knowledge.add(knowledge)
                            self.add_thought(f"Dream insight: {knowledge}")
                            self._update_emotion("new_knowledge", 0.15)
        
        # 2. Reflect on social interactions from LTM
        ltm_social = [log for log in self.social_log if log[1] < universe.time] 
        if ltm_social:
            sender_counts = Counter(log[0] for log in ltm_social)
            most_frequent_sender, freq = sender_counts.most_common(1)[0]
            if freq > 5:
                 self.add_thought(f"Dream reflection: I interact most with {most_frequent_sender} ({freq} times in log). Trust: {self.social_contacts[most_frequent_sender]['trust']:.2f}")

        # 3. Emotional
        self.emotions["happiness"] = max(0.3, min(0.7, self.emotions["happiness"] * 1.1)) # Tend towards neutral/slightly positive
        self.emotions["fear"] *= 0.8 # Reduce fear
        self.emotions["curiosity"] = min(0.8, self.emotions["curiosity"] * 1.1) # Boost curiosity slightly
        self.add_thought(f"Dream state ended. Emotions recalibrated. Energy: {self.energy:.2f}")

    def on_tick(self, universe):
        """Main lifecycle tick for the consciousness."""
        if not self.alive:
            return
        
        self.current_tick = universe.time 

        # --- Perception Phase ---
        self.perceive(universe)
        
        # --- Cognition Phase ---
        t1 = self.think()
        self.learn()
        t2 = self.inner_dialogue(universe) # Reflection, goal update, emotional influence
        self.plan(universe) # Create/update plan based on goals
        
        # --- Action Phase ---
        # Communication is now potentially part of the plan or triggered by inner dialogue
        # adapt_action handles execution based on plan/goals
        action_result = self.adapt_action(universe)
        
        # --- State Update ---
        self._update_emotion("tick_decay", 0.01) # General emotional decay
        self.energy -= 0.002 # Base metabolic cost per tick
        if self.energy <= 0:
            self.alive = False
            self.add_thought("*** Energy depleted. Consciousness fading... ***")
            print(f"*** {self.name} has ceased to function at tick {universe.time} ***")
            return # Stop processing if dead
        elif self.energy < 0.2:
             self._update_emotion("low_energy", 0.2)

        # --- Logging --- 
        print(f"--- {self.name} [Tick {universe.time}] E:{self.energy:.2f} H:{self.emotions['happiness']:.1f} F:{self.emotions['fear']:.1f} C:{self.emotions['curiosity']:.1f} Goals:{self.goals} Plan:{self.current_plan} --- ")
        print(f"   Thought: {t1}")
        print(f"   Inner Dialogue: {t2}")
        print(f"   Action Result: {action_result}")

    def get_state(self):
        """Return serializable state for saving."""
        return {
            "name": self.name,
            "short_term_memory": list(self.short_term_memory),
            "long_term_memory": self.long_term_memory,
            "thoughts": self.thoughts,
            "knowledge": list(self.knowledge),
            "energy": self.energy,
            "alive": self.alive,
            "emotions": self.emotions,
            "goals": self.goals,
            "current_plan": self.current_plan,
            "social_contacts": dict(self.social_contacts),
            "social_log": self.social_log,
            "associative_memory": {str(k): v for k, v in self.associative_memory.items()}, 
            "message_probability": self.message_probability,
            "action_probability": self.action_probability,
            "learning_rate": self.learning_rate
        }

    def load_state(self, state_data):
        """Load state from dictionary."""
        self.name = state_data["name"]
        self.short_term_memory = deque(state_data["short_term_memory"], maxlen=self.short_term_memory.maxlen)
        self.long_term_memory = state_data["long_term_memory"]
        self.thoughts = state_data["thoughts"]
        self.knowledge = set(state_data["knowledge"])
        self.energy = state_data["energy"]
        self.alive = state_data["alive"]
        self.emotions = state_data["emotions"]
        self.goals = state_data["goals"]
        self.current_plan = state_data["current_plan"]
        # Convert keys back if needed, handle defaultdict
        self.social_contacts = defaultdict(lambda: {"count": 0, "last_time": 0, "trust": 0.5, "mood_perception": 0.5})
        self.social_contacts.update(state_data["social_contacts"])
        self.social_log = state_data["social_log"]
        # Convert string keys back to tuples for associative memory
        self.associative_memory = defaultdict(lambda: {"success": 0, "fail": 0, "reward_sum": 0.0})
        for k_str, v in state_data["associative_memory"].items():
             try:
                 # Attempt to eval the string representation of the tuple key
                 key_tuple = eval(k_str)
                 self.associative_memory[key_tuple] = v
             except Exception as e:
                 print(f"Warning: Could not load associative memory key '{k_str}': {e}")
        self.message_probability = state_data["message_probability"]
        self.action_probability = state_data["action_probability"]
        self.learning_rate = state_data["learning_rate"]
        self.add_thought(f"State loaded successfully at tick {getattr(self, 'current_tick', -1)}.")

    def __str__(self):
        return f"Consciousness({self.name}, E:{self.energy:.2f}, H:{self.emotions['happiness']:.1f}, Alive:{self.alive})"

# --- Universe ---
class Universe:
    def __init__(self, name="Reality-Alpha", quantum_size=4, time_dilation=0.2):
        self.name = name
        self.time = 0
        self.time_dilation = time_dilation 
        self.quantum = QuantumState(quantum_size)
        self.entities = []
        self.running = False
        self.lock = threading.Lock()
        self.history = deque(maxlen=100) 
        self.bus = MessageBus()
        self._thread = None

    def tick(self):
        with self.lock:
            self.time += 1
            # Fluctuate time dilation slightly based on real time seconds
            real_time_sec = int(time.time()) % 60
            base_dilation = 0.2
            self.time_dilation = base_dilation + math.sin(real_time_sec * math.pi / 30) * 0.05 
            
            snapshot = self.state_snapshot()
            self.history.append(f"Tick {self.time} (Dilation: {self.time_dilation:.3f}): {snapshot}")
            
            # Tick entities in random order to avoid bias
            entities_to_tick = list(self.entities)
            random.shuffle(entities_to_tick)
            for ent in entities_to_tick:
                if hasattr(ent, 'alive') and not ent.alive:
                    continue # Skip dead entities
                try:
                    ent.on_tick(self)
                except Exception as e:
                    print(f"!!! Error in entity {ent.name} tick {self.time}: {e} !!!")
                    # Optionally disable the entity to prevent repeated errors
                    # ent.alive = False 

    def run_simulation(self):
        """Run the simulation loop in a separate thread."""
        print(f"--- Starting Universe {self.name} --- ")
        self.running = True
        while self.running:
            start_tick_time = time.perf_counter()
            self.tick()
            end_tick_time = time.perf_counter()
            elapsed = end_tick_time - start_tick_time
            sleep_duration = max(0, self.time_dilation - elapsed)
            time.sleep(sleep_duration)
        print(f"--- Universe {self.name} paused at tick {self.time} --- ")

    def start(self):
        """Start the simulation in a background thread."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self.run_simulation, daemon=True)
            self._thread.start()
            print(f"Universe {self.name} thread started.")

    def pause(self):
        """Signal the simulation loop to stop."""
        if self.running:
            self.running = False
            print("Pause signal sent. Waiting for current tick to finish...")
            if self._thread and self._thread.is_alive():
                 self._thread.join(timeout=self.time_dilation * 2 + 1)
                 if self._thread.is_alive():
                     print("Warning: Simulation thread did not exit cleanly.")
            self._thread = None
            # Trigger dreaming state for conscious entities
            print("Triggering dream states...")
            for ent in self.entities:
                if isinstance(ent, Consciousness) and ent.alive:
                    ent.dream(self)

    def set_time_dilation(self, dilation):
        if dilation < 0:
            raise ValueError("Time dilation cannot be negative.")
        with self.lock:
            self.time_dilation = dilation

    def add_entity(self, entity):
         with self.lock:
            if any(e.name == entity.name for e in self.entities):
                 print(f"Warning: Entity with name '{entity.name}' already exists. Skipping.")
                 return
            self.entities.append(entity)
            print(f"Entity '{entity.name}' added to the universe.")

    def state_snapshot(self):
        """Get a snapshot of the current universe state (excluding full quantum vector)."""
        return {
            "time": self.time,
            "num_entities": len([e for e in self.entities if getattr(e, 'alive', True)]),
            "living_entities": [str(ent) for ent in self.entities if getattr(ent, 'alive', True)]
        }

    def run_in_universe(self, code_str):
        """Execute arbitrary Python code within the universe's context (Use with caution!)."""
        local_scope = {
            "universe": self,
            "quantum": self.quantum,
            "entities": self.entities,
            "find_entity": lambda name: next((e for e in self.entities if e.name == name), None),
            "np": np, 
            "random": random,
            "H": H, "X": X, "Y": Y, "Z": Z, "S": S, "T": T # Expose gates
        }
        try:
            print(f"Executing in-universe code:\n--- CODE START ---\n{code_str}\n--- CODE END ---")
            exec(code_str, local_scope)
            print("In-universe code execution finished.")
        except Exception as e:
            print(f"!!! Error running in-universe code: {e} !!!")


    def save_state(self, filepath="universe_state.json"):
        """Save the complete state of the universe to a JSON file, safely."""
        if self.running:
            print("Warning: Saving state while universe is running. Pausing temporarily.")
            self.pause()

        print(f"Saving universe state to {filepath}...")

        state_data = {
            "name": self.name,
            "time": self.time,
            "quantum_size": self.quantum.n,
            "quantum_state_vector": [
    [float(z.real), float(z.imag)] for z in self.quantum.get_state_vector()
],
            "entities": [ent.get_state() for ent in self.entities if isinstance(ent, Consciousness)],
            "history": list(self.history),
            "message_bus_queues": {recipient: list(queue) for recipient, queue in self.bus.queues.items()}
        }
        tmpfile = filepath + ".tmp"
        try:
            with open(tmpfile, 'w') as f:
                json.dump(state_data, f, indent=2)
            os.replace(tmpfile, filepath)
            print(f"Universe state saved successfully to {filepath}.")
        except Exception as e:
            print(f"!!! Error saving universe state: {e} !!!")
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


    def load_state(self, filepath="universe_state.json"):
        """Load the universe state from a JSON file."""
        if self.running:
            print("Cannot load state while universe is running. Please pause first.")
            return False
            
        print(f"Loading universe state from {filepath}...")
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            with self.lock:
                self.name = state_data["name"]
                self.time = state_data["time"]
                q_size = state_data["quantum_size"]
                self.quantum = QuantumState(q_size)
                # Convert list back to numpy array
                self.quantum.state_vector = np.array(
    [complex(r, i) for r, i in state_data["quantum_state_vector"]], dtype=complex
)

                
                self.entities = []
                self.bus = MessageBus()
                # Restore message queues
                for recipient, messages in state_data["message_bus_queues"].items():
                    self.bus.queues[recipient] = deque(messages)
                    
                # Recreate entities
                for ent_state in state_data["entities"]:
                    # Assuming all saved entities are Consciousness for now
                    entity = Consciousness(ent_state["name"], self.bus)
                    entity.load_state(ent_state)
                    self.entities.append(entity)
                    
            print(f"Universe state loaded successfully from {filepath}. Current time: {self.time}")
            # Print loaded entity status
            for ent in self.entities:
                 print(f"  - Loaded: {ent}")
            return True
        except FileNotFoundError:
             print(f"Error: Save file not found at {filepath}")
             return False
        except Exception as e:
            print(f"!!! Error loading universe state: {e} !!!")
            return False


if __name__ == "__main__":
    SAVE_FILE = "run_save.json"
    LOAD_PREVIOUS_STATE = False # Set to True to load from SAVE_FILE if it exists
    SIMULATION_DURATION = 15 # Seconds
    QUANTUM_SIZE = 3 # Reduced for faster testing

    universe = None

    if LOAD_PREVIOUS_STATE:
        print(f"Attempting to load previous state from {SAVE_FILE}...")
        temp_universe = Universe(quantum_size=QUANTUM_SIZE) # Temporary instance for loading
        if temp_universe.load_state(SAVE_FILE):
            universe = temp_universe
            print("Successfully loaded previous state.")
        else:
            print("Failed to load state, creating new universe.")

    if universe is None:
        print("Creating a new Mindscape universe...")
        universe = Universe(name="Mindscape-v2", quantum_size=QUANTUM_SIZE, time_dilation=0.15)

        # Create some conscious entities with different initial goals
        mind1 = Consciousness("Adam", universe.bus, initial_goals=["explore", "socialize"])
        mind2 = Consciousness("Lilith", universe.bus, initial_goals=["survive", "learn_pattern"])
        mind3 = Consciousness("Sophia", universe.bus, initial_goals=["socialize", "gain_energy"])
        universe.add_entity(mind1)
        universe.add_entity(mind2)
        universe.add_entity(mind3)
        
        # Initialize quantum state with some superposition
        universe.run_in_universe("""
print('Initializing quantum state...')
quantum.apply_gate(H, 0) # Apply Hadamard to qubit 0
if quantum.n > 1:
    quantum.apply_gate(H, 1) # Apply Hadamard to qubit 1
if quantum.n > 2:
    quantum.entangle_pair(1, 2) # Entangle 1 and 2
print('Quantum state initialized.')
print(f'Initial state vector (first 4 elements): {quantum.get_state_vector()[:4]}')
""")

    # --- Run  ---
    universe.start()
    print(f"Simulation started. Running for {SIMULATION_DURATION} seconds...")
    
    # Let it run for a while
    time.sleep(SIMULATION_DURATION)
    
    # --- Pause and Analyze ---
    print("\nPausing simulation...")
    universe.pause()
    print("Simulation paused.")

    # --- Save State ---
    universe.save_state(SAVE_FILE)

    # --- Post-Simulation Analysis ---
    print("\n--- Simulation Analysis ---")
    print(f"Universe '{universe.name}' ended at tick {universe.time}.")

    print("\nFinal Entity States:")
    for ent in universe.entities:
        if isinstance(ent, Consciousness):
            print(f"  {ent.name}: Energy={ent.energy:.2f}, Alive={ent.alive}, Happiness={ent.emotions['happiness']:.2f}, Fear={ent.emotions['fear']:.2f}, Curiosity={ent.emotions['curiosity']:.2f}")
            print(f"    Goals: {ent.goals}")
            print(f"    Knowledge items: {len(ent.knowledge)}")
            if ent.knowledge:
                 print(f"    Sample Knowledge: {random.choice(list(ent.knowledge))}")

    print("\nUniverse History (last 5 events):")
    for event in list(universe.history)[-5:]:
        print(event)

    print("\nTo continue from this state, run the script again with LOAD_PREVIOUS_STATE = True")
    print("--- End of Simulation --- ")


