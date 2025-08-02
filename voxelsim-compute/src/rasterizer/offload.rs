use crate::rasterizer::{filter::FilterBindings, texture::TextureSet};

pub struct UpdaterManager<U: WorldUpdater> {
    channels: Vec<RenderChannel<U::ExternalBindings>>,
    updater: U,
    sender: crossbeam_channel::Sender<RenderTask<U::World>>,
}

pub struct RenderTask<W> {
    world: W,
}

pub struct ChannelBindings {
    pub rp_tex: TextureSet,
    pub rm_buf: FilterBindings,
}

pub struct RenderChannel<B> {
    pub bindings: ChannelBindings,
    pub external: B,
}

pub trait WorldUpdater {
    type World;
    type ExternalBindings;
    type Error: std::error::Error;

    fn create_channel(
        &mut self,
        channel_bindings: ChannelBindings,
    ) -> Result<RenderChannel<Self::ExternalBindings>, Self::Error>;

    fn update_world(
        &mut self,
        channel: &mut RenderChannel<Self::ExternalBindings>,
        world: &mut Self::World,
    ) -> Result<(), Self::Error>;
}
