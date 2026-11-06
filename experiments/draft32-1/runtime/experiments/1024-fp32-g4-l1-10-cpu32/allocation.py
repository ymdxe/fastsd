"""Allocate bounded, disjoint multicore CPU sets to 16 independent models."""
import collections
import math


def allocate(resource):
    by_core=collections.defaultdict(list)
    for row in resource['topology']:by_core[(row['socket'],row['core'])].append(row)
    budget=min(len(by_core),math.floor(resource['quota_cores']) if resource['quota_cores'] is not None else len(by_core))
    reserve=max(4,math.ceil(budget*.1))
    threads=max(1,min(4,(budget-reserve)//16))
    assert threads*16<=budget
    by_numa=collections.defaultdict(list)
    for siblings in by_core.values():
        row=dict(min(siblings,key=lambda r:r['cpu']));row['core_busy']=max(r['busy'] for r in siblings)
        by_numa[row['numa']].append(row)
    for cores in by_numa.values():cores.sort(key=lambda r:(r['core_busy'],r['cpu']))
    result=[];numas=sorted(by_numa)
    for worker in range(16):
        preferred=numas[worker%len(numas)]
        numa=preferred if len(by_numa[preferred])>=threads else max(numas,key=lambda n:len(by_numa[n]))
        assert len(by_numa[numa])>=threads,'Cannot fit a model inside one NUMA node'
        chosen=[by_numa[numa].pop(0) for _ in range(threads)]
        result.append({'worker':worker,'cpus':sorted(r['cpu'] for r in chosen),'numa':numa,'threads':threads,
                       'physical_cores':[(r['socket'],r['core']) for r in chosen]})
    cpus=[c for r in result for c in r['cpus']]
    assert len(cpus)==len(set(cpus)) and set(cpus)<=set(resource['allowed_logical_cpus'])
    assert len({tuple(c) for r in result for c in r['physical_cores']})==len(cpus)
    return result
