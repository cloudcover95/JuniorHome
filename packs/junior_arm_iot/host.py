"""TritARM host compact. Pair with quant.py. Not a Nintendo product."""
from __future__ import annotations
import json, re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from . import quant
AL,EQ,NE,LT,GE,GT,LE,NV=range(8)
OPS=("MOV","ADD","SUB","AND","ORR","EOR","LSL","LSR","LDR","STR","B","BL","CMP","SVC","LDRB","STRB")
OP={n:i for i,n in enumerate(OPS)}
REG={f"r{i}":i for i in range(16)}; REG.update(pc=15,lr=14,sp=13)
MMIO=0x40000000
PAD_LX,PAD_LY,PAD_BTNS=MMIO+0x20,MMIO+0x24,MMIO+0x28
UART_TX,GPIO_OUT,INTENT,THERMAL,FIELD=MMIO,MMIO+8,MMIO+0x40,MMIO+0x44,MMIO+0x48
VENDOR=(".xci",".nsp",".nca",".cia",".3ds",".nds",".gba",".z64")
ROSTER=("Vesper","Quill","Relay","Forge")
def u32(v): return v&0xFFFFFFFF
def encode(op,rd=0,rn=0,rm=0,imm=0,cond=AL,imm_mode=0):
    if op in (10,11): return u32((cond<<28)|(op<<24)|((imm_mode&1)<<23)|(imm&0x7FFFFF))
    return u32((cond<<28)|(op<<24)|((imm_mode&1)<<23)|(rd<<16)|(rn<<12)|(rm<<8)|(imm&255))
@dataclass(frozen=True)
class Instr:
    cond:int; op:int; rd:int; rn:int; rm:int; imm:int; imm_mode:int; raw:int=0
def decode(w):
    w=u32(w); op,cond,im=(w>>24)&15,(w>>28)&7,(w>>23)&1
    if op in (10,11):
        imm=w&0x7FFFFF
        if imm&0x400000: imm-=0x800000
        return Instr(cond,op,0,0,0,imm,im,w)
    return Instr(cond,op,(w>>16)&15,(w>>12)&15,(w>>8)&15,w&255,im,w)
@dataclass(frozen=True)
class Profile:
    name:str; ram:int; watts_hint:float; hz_hint:int; notes:str; pad_scale:float; deadzone:int
PROFILES={
 "generic_iot":Profile("generic_iot",65536,1.0,80000000,"default",1.0,4),
 "cortex_m0_iot":Profile("cortex_m0_iot",16384,0.5,48000000,"MCU",1.0,4),
 "arm7_class":Profile("arm7_class",32768,1.2,16000000,"envelope",1.0,6),
 "arm9_class":Profile("arm9_class",65536,2.0,66000000,"envelope",1.0,6),
 "a57_class":Profile("a57_class",262144,6.0,1000000000,"envelope. Not a Nintendo product.",0.85,8),
}
def get_profile(name): return PROFILES[(name or "generic_iot").lower()]
class MMIOMap:
    def __init__(self):
        self.uart_out=bytearray(); self.gpio_out=0; self.timer=0; self.halted=False
        self.pad_lx=0; self.pad_ly=0; self.pad_btns=0; self.intent=0; self.thermal=0; self.field=0; self.fwd=0
    def tick(self): self.timer=(self.timer+1)&0xFFFFFFFF
    def load32(self,a):
        a&=0xFFFFFFFF
        return {UART_TX:0,GPIO_OUT:self.gpio_out,PAD_LX:u32(self.pad_lx),PAD_LY:u32(self.pad_ly),PAD_BTNS:self.pad_btns,INTENT:u32(self.intent),THERMAL:self.thermal,FIELD:self.field,MMIO+0x10:self.timer}.get(a,0)
    def store32(self,a,v):
        a,v=a&0xFFFFFFFF,v&0xFFFFFFFF
        if a==UART_TX: self.uart_out.append(v&255); self.fwd+=1
        elif a==GPIO_OUT: self.gpio_out=v
        elif a==PAD_LX: self.pad_lx=v-0x100000000 if v&0x80000000 else v; self.fwd+=1
        elif a==PAD_LY: self.pad_ly=v-0x100000000 if v&0x80000000 else v
        elif a==PAD_BTNS: self.pad_btns=v
        elif a==INTENT: self.intent=v-0x100000000 if v&0x80000000 else v
        elif a==THERMAL: self.thermal=v
        elif a==FIELD: self.field=v
        elif a==MMIO+0x14: self.halted=bool(v)
class Memory:
    def __init__(self,size):
        self.size=size; self.buf=bytearray(size); self.mmio=MMIOMap(); self.writes=0
    def load32(self,a):
        a&=0xFFFFFFFC
        if a>=MMIO: return self.mmio.load32(a)
        if a+3>=self.size: return 0
        b=self.buf; return b[a]|(b[a+1]<<8)|(b[a+2]<<16)|(b[a+3]<<24)
    def store32(self,a,v):
        a&=0xFFFFFFFC; self.writes+=1
        if a>=MMIO: self.mmio.store32(a,v); return
        if a+3>=self.size: return
        v&=0xFFFFFFFF; self.buf[a:a+4]=bytes((v&255,(v>>8)&255,(v>>16)&255,(v>>24)&255))
    def load8(self,a):
        a&=0xFFFFFFFF
        if a>=MMIO: return self.mmio.load32(a&0xFFFFFFFC)&255
        return self.buf[a] if a<self.size else 0
    def store8(self,a,v):
        a&=0xFFFFFFFF; self.writes+=1
        if a>=MMIO: self.mmio.store32(a&0xFFFFFFFC,v&255); return
        if a<self.size: self.buf[a]=v&255
    def write_bytes(self,a,data):
        for i,b in enumerate(data): self.store8(a+i,b)
class Cpu:
    def __init__(self,mem):
        self.mem=mem; self.r=[0]*16; self.n=0; self.z=1; self.cycles=0; self.halted=False; self.hist=[0]*16
    def set_nz(self,v):
        v=u32(v); self.n=1 if v&0x80000000 else 0; self.z=int(v==0); return v
    def ok(self,c):
        return (True,self.z,not self.z,self.n,not self.n,(not self.z)and(not self.n),self.z or self.n,False)[c]
    def step(self):
        if self.halted or self.mem.mmio.halted: self.halted=True; return
        pc=self.r[15]&0xFFFFFFFC; ins=decode(self.mem.load32(pc))
        cond,op,rd,rn,rm,imm,im=ins.cond,ins.op,ins.rd,ins.rn,ins.rm,ins.imm,ins.imm_mode
        self.r[15]=u32(pc+4); self.cycles+=1; self.mem.mmio.tick()
        if not self.ok(cond): return
        self.hist[op]+=1; src=imm if im else self.r[rm]
        if op==0: self.r[rd]=self.set_nz(src)
        elif op==1: self.r[rd]=self.set_nz(self.r[rn]+src)
        elif op==2: self.r[rd]=self.set_nz(self.r[rn]-src)
        elif op==3: self.r[rd]=self.set_nz(self.r[rn]&src)
        elif op==4: self.r[rd]=self.set_nz(self.r[rn]|src)
        elif op==5: self.r[rd]=self.set_nz(self.r[rn]^src)
        elif op==6: self.r[rd]=self.set_nz(self.r[rn]<<(src&31))
        elif op==7: self.r[rd]=self.set_nz((self.r[rn]&0xFFFFFFFF)>>(src&31))
        elif op==8: self.r[rd]=self.mem.load32(u32(self.r[rn]+(imm if im else self.r[rm])))
        elif op==9: self.mem.store32(u32(self.r[rn]+(imm if im else self.r[rm])),self.r[rd])
        elif op==10: self.r[15]=u32(self.r[15]+imm)
        elif op==11: self.r[14]=self.r[15]; self.r[15]=u32(self.r[15]+imm)
        elif op==12: self.set_nz(self.r[rn]-src)
        elif op==13:
            if (imm if im else src)==0: self.halted=True; self.mem.mmio.halted=True
        elif op==14: self.r[rd]=self.mem.load8(u32(self.r[rn]+(imm if im else self.r[rm])))
        elif op==15: self.mem.store8(u32(self.r[rn]+(imm if im else self.r[rm])),self.r[rd])
        self.r[rd]=u32(self.r[rd]); self.r[15]=u32(self.r[15])
def _src_tok(tok,equs):
    t=tok.rstrip(",")
    if t.lower() in REG: return 0,REG[t.lower()],0
    if t.startswith("#"): t=t[1:]
    if t.lower() in equs: return 1,0,equs[t.lower()]&255
    return 1,0,(int(t,16) if t.lower().startswith("0x") else int(t))&255
def assemble(src,origin=0):
    labels,equs,rows,addr={},{},[],origin
    for raw in src.splitlines():
        line=raw.split(";")[0].strip()
        if not line: continue
        if line.upper().startswith("EQU "):
            _,n,v=line.replace(","," ").split(); equs[n.lower()]=int(v.lstrip("#"),0) if False else int(v[1:],16) if v.lower().startswith("#0x") or v.lower().startswith("0x") else int(v.lstrip("#")); continue
        if line.endswith(":"): labels[line[:-1].lower()]=addr; continue
        rows.append((addr,line)); addr+=4
    out=bytearray()
    for addr,line in rows:
        parts=[p for p in re.split(r"[\s,]+",line) if p]; op_tok=parts[0].upper(); cond=AL
        for suf,c in (("EQ",EQ),("NE",NE),("LT",LT),("GE",GE),("GT",GT),("LE",LE)):
            if op_tok.endswith(suf) and op_tok[:-len(suf)] in OP: cond=c; op_tok=op_tok[:-len(suf)]; break
        if op_tok=="BNE": op_tok,cond="B",NE
        if op_tok=="BGT": op_tok,cond="B",GT
        if op_tok=="BLT": op_tok,cond="B",LT
        if op_tok=="BGE": op_tok,cond="B",GE
        if op_tok=="BLE": op_tok,cond="B",LE
        if op_tok=="BEQ": op_tok,cond="B",EQ
        op=OP[op_tok]; args=parts[1:]
        if op in (10,11):
            t=args[0].lower(); imm=labels[t]-(addr+4) if t in labels else int(args[0]); out.extend(encode(op,imm=imm,cond=cond).to_bytes(4,"little")); continue
        if op==13: out.extend(encode(op,imm=_src_tok(args[0] if args else "0",equs)[2],cond=cond,imm_mode=1).to_bytes(4,"little")); continue
        if op==12:
            im,rm,imm=_src_tok(args[1],equs); out.extend(encode(op,rn=REG[args[0].lower().rstrip(",")],rm=rm,imm=imm,imm_mode=im,cond=cond).to_bytes(4,"little")); continue
        if op in (8,9,14,15):
            rd=REG[args[0].lower().rstrip(",")]; bits="".join(args[1:]).strip("[]").split(","); rn=REG[bits[0].lower()]
            if len(bits)==1: out.extend(encode(op,rd=rd,rn=rn,imm=0,imm_mode=1,cond=cond).to_bytes(4,"little"))
            else:
                im,rm,imm=_src_tok(bits[1],equs); out.extend(encode(op,rd=rd,rn=rn,rm=rm,imm=imm,imm_mode=im,cond=cond).to_bytes(4,"little"))
            continue
        if op==0:
            rd=REG[args[0].lower().rstrip(",")]; im,rm,imm=_src_tok(args[1],equs); out.extend(encode(op,rd=rd,rm=rm,imm=imm,imm_mode=im,cond=cond).to_bytes(4,"little")); continue
        rd=REG[args[0].lower().rstrip(",")]; rn=REG[args[1].lower().rstrip(",")]; im,rm,imm=_src_tok(args[2],equs)
        out.extend(encode(op,rd=rd,rn=rn,rm=rm,imm=imm,imm_mode=im,cond=cond).to_bytes(4,"little"))
    return bytes(out)
def hist_features(hist):
    total=float(sum(hist) or 1); groups=((0,1,2),(3,4,5),(6,7),(8,14),(9,15),(10,11),(12,),(13,))
    return [sum(hist[i] for i in g)/total for g in groups]
def score_intent(hist,writes=0,cycles=1):
    feat=hist_features(hist); trit=1 if feat[5]>feat[4] else (-1 if feat[4]>0.25 else 0)
    return type("I",(),{"trit":trit,"label":{-1:"retreat",0:"hold",1:"approach"}[trit],"thermal":min(255,sum(hist)+writes),"features":feat})()
def retarget_pad(lx,ly,buttons,profile="generic_iot",roster="Forge",trit=0):
    prof=profile if isinstance(profile,Profile) else get_profile(profile); who=roster if roster in ROSTER else "Forge"
    def axis(raw):
        v=raw*prof.pad_scale
        if abs(v)<prof.deadzone: return 0.0
        v=v/127.0; return -1.0 if v<-1 else 1.0 if v>1 else v
    return type("P",(),{"lx":axis(lx),"ly":axis(ly),"buttons":buttons&0xFFFF,"roster":who,"profile":prof.name,"trit":int(trit)})()
def as_frameforge(pad):
    return {"roster":pad.roster,"lx":pad.lx,"ly":pad.ly,"buttons":pad.buttons,"trit":pad.trit,"profile":pad.profile,"role":"cpu_intent_pad","legal":"not a nintendo product"}
def classify_bytes(data,name=""):
    suf=Path(name).suffix.lower() if name else ""; magic=data[:4]
    info={"name":name,"suffix":suf,"size":len(data),"sig":int.from_bytes(magic or b"\0\0\0\0","big"),"decrypt":False,"legal":"not a nintendo product"}
    if suf in (".tas",".s",".asm",".txt") or b"MOV" in data[:200] or b"SVC" in data[:200]: info.update(kind="tritarm",mode="execute"); return info
    if magic==b"PFS0": info.update(kind="nsp_pfs0",mode="classify"); return info
    if magic[:3]==b"NCA": info.update(kind="nca_magic",mode="classify"); return info
    if suf==".gba": info.update(kind="gba_header",mode="header",title=data[0xA0:0xAC].split(b"\0")[0].decode("latin1","ignore")); return info
    if suf in VENDOR: info.update(kind="vendor_class",mode="classify"); return info
    info.update(kind="raw",mode="execute"); return info
class Machine:
    def __init__(self,profile="generic_iot",roster="Forge"):
        self.profile=profile if isinstance(profile,Profile) else get_profile(profile); self.roster=roster
        self.mem=Memory(self.profile.ram); self.cpu=Cpu(self.mem); self.intent=score_intent([0]*16)
        self.last_format={"kind":"empty","mode":"idle","decrypt":False,"legal":"not a nintendo product"}
        self.last_sis=None; self.last_xr=None; self.last_crispy=None; self.last_engine=None
        self.mmio_trits=[]; self.mmio_packed=b""; self.mmio_scale=0.0
        self._last_pad=retarget_pad(0,0,0,self.profile,roster); self.palace_n=0
    def load(self,blob,addr=0):
        self.mem.write_bytes(addr,blob); self.cpu.r[15]=addr; self.cpu.r[13]=self.profile.ram-16
    def load_asm(self,src,addr=0):
        blob=assemble(src,addr); self.load(blob,addr); return blob
    def load_path(self,path):
        p=Path(path); data=p.read_bytes(); info=classify_bytes(data,p.name); self.last_format=info
        if info["mode"]=="execute":
            if p.suffix.lower() in (".tas",".s",".asm",".txt"): self.load_asm(data.decode("utf-8")); return
            self.load(data)
    def _sample(self):
        self.intent=score_intent(self.cpu.hist,self.mem.writes,max(1,self.cpu.cycles)); mm=self.mem.mmio
        mm.store32(INTENT,self.intent.trit); mm.store32(THERMAL,self.intent.thermal)
        self._last_pad=retarget_pad(mm.pad_lx,mm.pad_ly,mm.pad_btns,self.profile,self.roster,self.intent.trit)
        feat=quant.mmio_features(mm.pad_lx,mm.pad_ly,mm.pad_btns,len(mm.uart_out),mm.gpio_out,mm.fwd,self.cpu.cycles)
        self.mmio_trits,self.mmio_scale=quant.quant_vec(feat); self.mmio_packed=quant.pack_trits(self.mmio_trits)
        mm.store32(FIELD,(self.mmio_trits[0]+1)|((self.mmio_trits[1]+1)<<8)); self.palace_n+=1
    def run(self,steps=10000):
        n=0
        while n<steps and not self.cpu.halted:
            self.cpu.step(); n+=1
            if n%64==0: self._sample()
        self._sample(); return self.snapshot()
    def snapshot(self):
        mm=self.mem.mmio
        return {"profile":self.profile.name,"cycles":self.cpu.cycles,"halted":self.cpu.halted,"pc":self.cpu.r[15],"regs":list(self.cpu.r),"uart":bytes(mm.uart_out).decode("latin1"),"gpio_out":mm.gpio_out,"intent":{"trit":self.intent.trit,"label":self.intent.label,"thermal":self.intent.thermal,"features":self.intent.features},"pad":as_frameforge(self._last_pad),"switch":{"fwd":mm.fwd,"drops":0,"pending":0},"palace":self.palace_n,"quant":{"trits":list(self.mmio_trits),"scale":self.mmio_scale,"packed":list(self.mmio_packed),"scheme":"absmean_b1.58"},"format":self.last_format,"sis":self.last_sis,"xr":self.last_xr,"crispy":self.last_crispy,"engine":self.last_engine,"legal":"not a nintendo product"}
    def ingest_crispy(self,packet):
        lx=int(packet.get("lx",packet.get("x",0))); ly=int(packet.get("ly",packet.get("y",0))); btns=int(packet.get("buttons",0))
        zone=str(packet.get("gaze_zone",packet.get("zone",""))); btns|={"TOP_LEFT":1,"TOP_RIGHT":2,"BOTTOM_LEFT":4,"BOTTOM_RIGHT":8}.get(zone,0)
        self.mem.mmio.store32(PAD_LX,lx); self.mem.mmio.store32(PAD_LY,ly); self.mem.mmio.store32(PAD_BTNS,btns)
        self.last_crispy={"lx":lx,"ly":ly,"buttons":btns,"zone":zone,"source":"crispy-mouse","legal":"not a nintendo product"}; self._sample(); return self.last_crispy
    def xr_beta(self,extra=None):
        extra=extra or {}; snap=self.snapshot()
        self.last_xr={"kind":"xr_beta","pos":extra.get("pos",[0.0,0.0,0.0]),"quat":extra.get("quat",[0.0,0.0,0.0,1.0]),"pad":snap["pad"],"trit":snap["intent"]["trit"],"source":extra.get("source","guest"),"hz_hint":72,"unreal":False,"legal":"not a nintendo product"}; return self.last_xr
    def route_engine(self,name,payload=None):
        key=(name or "fieldcore").lower(); dest={"omega":"omega","blender":"omega","juniorllm":"juniorllm","llm":"juniorllm","agi":"agi_sdk"}.get(key,"fieldcore")
        self.last_engine={"engine":dest,"asked":name,"train":False,"mode":"input_x_function"}; return self.last_engine
    def intent_line(self): return json.dumps(self.snapshot()["pad"],separators=(",",":"))
MOV,ADD,SUB,AND,ORR,EOR,LSL,LSR,LDR,STR,B,BL,CMP,SVC,LDRB,STRB=range(16)
PC,LR,SP=15,14,13
def absmean_trit(xs,theta=0.08):
    if not xs: return 0
    s=sum(abs(x) for x in xs)/len(xs)
    if s<=1e-12: return 0
    acc=sum(xs)/(s*len(xs)); return 1 if acc>=theta else -1 if acc<=-theta else 0
class Palace:
    def __init__(self,cap=4096): self.ring=deque(); self.cap=cap
    def remember(self,tr):
        if len(self.ring)>=self.cap: self.ring.popleft()
        self.ring.append(tr)
def sis_commit(palace,meta=None): return {"backend":"palace_local","n":len(getattr(palace,"ring",[])),"meta":meta or {},"msis":None}
def apply_crispy(machine,packet): return machine.ingest_crispy(packet)
def xr_pose(machine,extra=None): return machine.xr_beta(extra)
def engine_route(name,payload=None): return Machine().route_engine(name,payload)
