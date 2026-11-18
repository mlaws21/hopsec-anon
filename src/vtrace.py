from __future__ import annotations
import json, hashlib, time, copy
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .task import TaskChain
    
from . import timing

import threading

_seen, _seen_lock = {True: set(), False: set()}, threading.Lock()

USERS = ["darrow", "sevro", "mustang", "cassius"]
EPSILON = 1 
    
# Generate a public-private keypair using the ed25519 algorithm.
def generate_ed25519_keypair() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:

    # Generate Ed25519 Private Key (for signing)
    ed25519_private_key = ed25519.Ed25519PrivateKey.generate()

    # Extract Ed25519 Public Key
    ed25519_public_key = ed25519_private_key.public_key()

    return ed25519_private_key, ed25519_public_key

# hashes are returned as hexstrings
def H(m) -> str: 
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()

# signatures are returned as hexstrings (remember to encode as bytes before verifying)
def sign(m: dict, sk: ed25519.Ed25519PrivateKey) -> str:
    return sk.sign(H(m).encode()).hex()

# verifies signature under a given pk
def verify(m: dict, signature: str, pk: ed25519.Ed25519PublicKey) -> bool:
    try:
        pk.verify(bytes.fromhex(signature), H(m).encode())
        return True
    except (InvalidSignature, ValueError):
        return False

# remaining_work is the Q_r the receiver gets, at first step its Q from the user
# sender_work is the sender's newest progress entry {step_id, description, result}
def build_message_dict(qid: str, ts: float, step_num: int, deadline: float, sender: str, receiver: str, sender_work: dict | None, remaining_work: str | None, allowed_agents: list, prev_hash: str | None) -> dict:
    m = dict(qid=qid, ts=ts, step_num=step_num, deadline=deadline, sender=sender, receiver=receiver, sender_work=sender_work, remaining_work=remaining_work, allowed_agents=allowed_agents, prev_hash=prev_hash)
    return m

def agent_name(aid) -> str:
    aid = str(aid)
    return aid if aid.startswith("agent-") else f"agent-{aid}"

def user_name(uid: str) -> str:
    return uid if uid.startswith("user-") else f"user-{uid}"

# Signs and appends one message.
def add_step(task_chain: TaskChain, sender: str, receiver: str, remaining_work: str | None, sk: ed25519.Ed25519PrivateKey | None, most_recent_trace: dict | None = None) -> dict:

    trace = task_chain.vet
    prev_hash = None
    if sk is not None and trace:
        with timing.timed("crypto_sign"):
            prev_hash = H(trace[-1][0])
    m = build_message_dict(
        qid=task_chain.qid, ts=time.time(), step_num=len(trace),
        deadline=task_chain.deadline, sender=sender, receiver=receiver,
        sender_work=task_chain.pop_pending_work(), remaining_work=remaining_work,
        allowed_agents=list(task_chain.allowed_agents), prev_hash=prev_hash)
    sig = ""
    if sk is not None:
        with timing.timed("crypto_sign"):
            sig = sign(m, sk)
    trace.append((m, sig))
    if most_recent_trace is not None:
        with timing.timed("trace_log"):
            if task_chain.qid not in most_recent_trace:
                most_recent_trace[task_chain.qid] = []
            most_recent_trace[task_chain.qid].append(copy.deepcopy(trace))
    return m

#  VET -> AgentNet views
def is_user(name: str) -> bool:
    return name.startswith("user-")

# extracts the work done so far in AgentNet compatible form (allowed to be edited, for attack)
def work_refs(vet: list, pending) -> list:
    refs = [m["sender_work"] for m, _ in vet if m["sender_work"] is not None]
    if pending is not None:
        refs.append(pending[1])
    return refs

# extracts full progress in AgentNet compatible form
def progress_from_vet(vet: list, pending, id_of) -> list:
    entries = [{"agent_id": id_of(m["sender"]), **m["sender_work"]}
               for m, _ in vet if m["sender_work"] is not None]
    if pending is not None:
        entries.append({"agent_id": pending[0], **pending[1]})
    return entries


# extract remaining work (or nothing if main task)
def received_remaining_work(vet: list) -> str:
    m = vet[-1][0]
    return "" if is_user(m["sender"]) else m["remaining_work"]


def major_problem_from_vet(vet: list) -> str:
    return vet[0][0]["remaining_work"]


def final_result_from_vet(vet: list) -> str:
    if not vet:
        return ""
    m = vet[-1][0]
    if not is_user(m["receiver"]) or m["sender_work"] is None:
        return ""
    return m["sender_work"]["result"]

# Checks the chain in both modes against one receipt time; each mode's
# verdict is independent (own duplicate set) and timed into its own bucket.
def check_incoming_both(trace: list, me: str, pks: dict) -> dict:
    now = time.time()
    return {"all": check_incoming(trace, me, pks, verify_all=True, now=now),
            "one": check_incoming(trace, me, pks, verify_all=False, now=now)}

# verify_all: every message's signature + hash links; else only the newest
# sender's. now: the receipt time (default: when called).
def check_incoming(trace: list, me: str, pks: dict, verify_all: bool=True, now: float | None=None) -> tuple[bool, str]:
    with timing.timed("verify_all" if verify_all else "verify_one"):
        return _check_incoming(trace, me, pks, verify_all, time.time() if now is None else now)

def _check_incoming(trace: list, me: str, pks: dict, verify_all: bool, now: float) -> tuple[bool, str]:
    m, sig = trace[-1]
    # step 1: sender verification
    if not verify_all:
        pk = pks.get(m["sender"])
        if pk is None or not verify(m, sig, pk):
            return False, "bad sender signature"
    else:
        for i, (t, sigt) in enumerate(trace):
            pkt = pks.get(t["sender"])
            if pkt is None or not verify(t, sigt, pkt):
                return False, f"bad sender signature at step {i}"
            
            # vrfy hashchain too
            expected_prev = H(trace[i - 1][0]) if i > 0 else None
            if t["prev_hash"] != expected_prev:
                return False, f"broken hash link at step {i}"
            
        
    
    # step 2: agent identities
    if m["receiver"] != me:
        return False, "wrong receiver"
    
    # if len trace == 1 that is from the user
    if len(trace) > 1:
        if m["sender"] not in m["allowed_agents"]:
            return False, "sender not allowed"
    else:
        # this could read from the PKI
        if m["sender"] not in [user_name(x) for x in USERS]:
            return False, "user not authorized"
    
    if me not in m["allowed_agents"]:
        return False, "receiver not allowed"
    
    # step 3: user parameters (do i need this, no but why not?)
    if len(trace) > 1: # otherwise this is just repeated work
        user_m, user_sig = trace[0]
        user_pk = pks.get(user_m["sender"])
        
        if user_pk is None or not verify(user_m, user_sig, user_pk):
            return False, "someone tampered with main task"
        
        if user_m["deadline"] != m["deadline"]:
            return False, "deadline changed"
        
        if user_m["allowed_agents"] != m["allowed_agents"]:
            return False, "allowed agents changed"
        
        if user_m["qid"] != m["qid"]:
            return False, "qid changed"
    
        if m["sender"] not in [user_name(x) for x in USERS]:
            return False, "user not authorized"
    
    # step 4: deadline and ordering
    if now > m["deadline"] or m["ts"] > m["deadline"]:
        return False, "deadline expired"
    
    if abs(m["ts"] - now) > EPSILON:
        return False, "invalid time"
    
    if m["step_num"] != len(trace) - 1:
        return False, "invalid step number"
    
    # step 5: duplication detection
    key = (me, m["qid"], m["step_num"])
    
    with _seen_lock:
        if key in _seen[verify_all]:
            return False, "duplicate message"
        _seen[verify_all].add(key)
    
    return True, ""
    
    