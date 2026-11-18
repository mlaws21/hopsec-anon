import argparse

from common import load_checkpoint, router_entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="training_state.pkl or the save dir holding it")
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint)
    for aid in sorted(state["agents"]):
        entries = router_entries(state["agents"][aid])
        print(f"Agent{aid}: {len(entries)} unique entries")
        for i, text in enumerate(entries, 1):
            print(f"  {i}. {text}")


if __name__ == "__main__":
    main()
