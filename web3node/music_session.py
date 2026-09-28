from music_wire import wire
def session(name="junior-deck"):
    w = wire()
    return {"name": name, "engine": "JuniorDeck", "daw": "ardour-host",
            "drums": "hydrogen-host", "edit": "audacity-bounce",
            "wire": w["graph"], "live": False, "pad": w["deck"]["pad"]}
