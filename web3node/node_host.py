"""Node hosts a default Home stack. Hits are a counter, not a file."""
STACK = ("gaia", "deck", "field", "koirig", "home")
def host(user):
    return {"user": user, "stack": list(STACK), "personalized": True, "disk": False}
def count(hits):
    tallies = {name: 0 for name in STACK}
    for name in hits:
        if name in tallies:
            tallies[name] += 1
    return {"tallies": tallies, "n": sum(tallies.values()), "disk": False,
            "ram_hold": True, "network": False}
