use {
    crate::{layout::*, ranges::*},
    gfx_hal::{
        device::OutOfMemory,
        pso::{AllocationError, DescriptorPool as _},
        Backend, Device,
    },
    smallvec::{smallvec, SmallVec},
    std::{
        cmp::{max, min},
        collections::{HashMap, VecDeque},
    },
};

const MIN_SETS: u32 = 64;

pub struct DescriptorSet<B: Backend> {
    raw: B::DescriptorSet,
    pool: u64,
}

struct Allocation<B: Backend> {
    sets: Vec<B::DescriptorSet>,
    pools: Vec<u64>,
}

struct DescriptorPool<B: Backend> {
    raw: B::DescriptorPool,
    size: u32,
    free: u32,
}

unsafe fn allocate_from_pool<B: Backend>(
    raw: &mut B::DescriptorPool,
    layout: &B::DescriptorSetLayout,
    count: u32,
    allocation: &mut Vec<B::DescriptorSet>,
) -> Result<(), OutOfMemory> {
    let sets_were = allocation.len();
    raw.allocate_sets(std::iter::repeat(layout).take(count as usize), allocation)
        .map_err(|err| match err {
            AllocationError::OutOfHostMemory => OutOfMemory::OutOfHostMemory,
            AllocationError::OutOfDeviceMemory => OutOfMemory::OutOfDeviceMemory,
            err => {
                panic!("Unexpected error: {}", err);
            }
        })?;
    assert_eq!(allocation.len(), sets_were + count as usize);
    Ok(())
}

struct DescriptorBucket<B: Backend> {
    pools_offset: u64,
    pools: VecDeque<DescriptorPool<B>>,
}

impl<B> DescriptorBucket<B>
where
    B: Backend,
{
    fn new() -> Self {
        DescriptorBucket {
            pools_offset: 0,
            pools: VecDeque::new(),
        }
    }

    unsafe fn allocate(
        &mut self,
        device: &impl Device<B>,
        layout: &DescriptorSetLayout<B>,
        mut count: u32,
        allocation: &mut Allocation<B>,
    ) -> Result<(), OutOfMemory> {
        if count == 0 {
            return Ok(());
        }

        for (index, pool) in self.pools.iter_mut().enumerate().rev() {
            if pool.free == 0 {
                continue;
            }

            let allocate = min(pool.free, count);
            allocate_from_pool::<B>(&mut pool.raw, layout.raw(), allocate, &mut allocation.sets)?;
            allocation.pools.extend(
                std::iter::repeat(index as u64 + self.pools_offset).take(allocate as usize),
            );
            count -= allocate;
            pool.free -= allocate;

            if count == 0 {
                return Ok(());
            }
        }

        if count > 0 {
            let size = max(MIN_SETS, (count - 1).next_power_of_two() * 2);
            let mut raw = device.create_descriptor_pool(size as usize, &layout.ranges())?;
            allocate_from_pool::<B>(&mut raw, layout.raw(), count, &mut allocation.sets)?;
            allocation.pools.extend(
                std::iter::repeat(self.pools.len() as u64 + self.pools_offset).take(count as usize),
            );
            self.pools.push_back(DescriptorPool {
                raw,
                size,
                free: size - count,
            });
        }

        Ok(())
    }

    unsafe fn free(&mut self, sets: impl IntoIterator<Item = B::DescriptorSet>, pool: u64) {
        let pool = &mut self.pools[(pool - self.pools_offset) as usize];
        pool.raw.free_sets(sets);
    }

    unsafe fn cleanup(&mut self, device: &impl Device<B>) {
        while let Some(pool) = self.pools.pop_front() {
            if pool.free < pool.size {
                self.pools.push_front(pool);
                break;
            }
            device.destroy_descriptor_pool(pool.raw);
            self.pools_offset += 1;
        }
    }
}

pub struct DescriptorAllocator<B: Backend> {
    buckets: HashMap<DescriptorRanges, DescriptorBucket<B>>,
    allocation: Allocation<B>,
    relevant: relevant::Relevant,
}

impl<B> DescriptorAllocator<B>
where
    B: Backend,
{
    pub fn new() -> Self {
        DescriptorAllocator {
            buckets: HashMap::new(),
            allocation: Allocation {
                sets: Vec::new(),
                pools: Vec::new(),
            },
            relevant: relevant::Relevant,
        }
    }

    pub unsafe fn dispose(mut self, device: &impl Device<B>) {
        self.cleanup(device);
        if self.buckets.values().any(|b| !b.pools.is_empty()) {
            log::error!("Not all descriptor sets were freed");
        }
        self.relevant.dispose();
    }

    pub unsafe fn allocate(
        &mut self,
        device: &impl Device<B>,
        layout: &DescriptorSetLayout<B>,
        count: u32,
        extend: &mut impl Extend<DescriptorSet<B>>,
    ) -> Result<(), OutOfMemory> {
        let layout_ranges = layout.ranges();
        let bucket = self
            .buckets
            .entry(layout_ranges)
            .or_insert_with(|| DescriptorBucket::new());
        match bucket.allocate(device, layout, count, &mut self.allocation) {
            Ok(()) => {
                extend.extend(
                    Iterator::zip(
                        self.allocation.pools.drain(..),
                        self.allocation.sets.drain(..),
                    )
                    .map(|(pool, set)| DescriptorSet { raw: set, pool }),
                );
                Ok(())
            }
            Err(err) => {
                // Free sets allocated so far.
                let mut last = None;
                for (index, pool) in self.allocation.pools.drain(..).enumerate().rev() {
                    match last {
                        Some(last) if last != pool => {
                            bucket.free(self.allocation.sets.drain(index + 1..), last);
                        }
                        _ => last = Some(pool),
                    }
                }
                Err(err)
            }
        }
    }

    pub unsafe fn free(
        &mut self,
        sets: impl IntoIterator<Item = DescriptorSet<B>>,
        layout: &DescriptorSetLayout<B>,
    ) {
        let layout_ranges = layout.ranges();
        let bucket = self
            .buckets
            .get_mut(&layout_ranges)
            .expect("Sets with specified layout must be allcoated from this pool");

        let mut free: Option<(u64, SmallVec<[B::DescriptorSet; 32]>)> = None;
        for set in sets {
            match &mut free {
                Some((pool, sets)) if *pool == set.pool => {
                    sets.push(set.raw);
                }
                Some((pool, sets)) => {
                    bucket.free(sets.drain(), *pool);
                    *pool = set.pool;
                    sets.push(set.raw);
                }
                slot @ None => {
                    *slot = Some((set.pool, smallvec![set.raw]));
                }
            }
        }
    }

    pub unsafe fn cleanup(&mut self, device: &impl Device<B>) {
        self.buckets
            .values_mut()
            .for_each(|bucket| bucket.cleanup(device));
    }
}
