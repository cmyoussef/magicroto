# dummy function to update with external execution
card_class = None

try:
    import nuke
    from simple_executer import card
    card_class = card.SetupCards
except:
    pass

def run_external(cmd, prio='low', requirements=None, **kyeargs):
    requirements = requirements or {'gpu': True, 'nuke':True, 'ram': 64, 'cpu': 4}
    if card_class is None:

        nuke.tprint("This advanced feature enables you to utilize external machines, "
            "either through a render farm or by running advanced code.")
        return
    jobName = kyeargs.get('jobName', '')
    cardName = kyeargs.get('cardName', '')

    card = card_class({'cmd':cmd}, f'SD-{cardName}', f'NukeSD-{jobName}')
    tractor = card.dispatch_job()
    nuke.tprint(f"Job Submitted : {tractor}")
    return tractor