#include "r_fs.h"
#include "fs/vfs.h"
#include "fs/xv6fs/defs.h"
#include "defs.h"
#include "fs/xv6fs/fs.h"
#include "fs/xv6fs/file.h"
#include "param.h"
#include "stat.h"
#include "buf.h"

static struct filesystem_operations xv6fs_ops = {
    .mount = xv6_mount,
    .umount = xv6_umount,
    .alloc_inode = xv6_alloc_inode,
    .write_inode = xv6_write_inode,
    .release_inode = xv6_release_inode,
    .free_inode = xv6_free_inode,
    .trunc = xv6_trunc,
    .open = xv6_open,
    .close = xv6_close,
    .read = xv6_read,
    .write = xv6_write,
    .create = xv6_create,
    .link = xv6_link,
    .unlink = xv6_unlink,
    .dirlookup = xv6_dirlookup,
    .release_dentry = xv6_release_dentry,
    .isdirempty = xv6_isdirempty,
    .init = xv6_init,
    .update_inode = xv6_update_inode,
};

struct filesystem_type xv6fs = {
.type = "xv6fs",
.op = &xv6fs_ops ,
};

void init(){

  root=kalloc();
  if(!root){
    return;
  }
  root->type=&xv6fs;
  root->op=&xv6fs_ops;
  root->op->init();
  root->root = inget(1, 1);
  root->parent = 0;
  root->mountpoint = 0;
  for(int i=0;i<MAXMNT;i++){
    root->mounts[i]=0;
  }
  root->private = &sb;
 // printf("vfs init\n");
}

static void
bzero(int dev, int bno)
{
  struct buf *bp;

  bp = bread(dev, bno);
  memset(bp->data, 0, BSIZE);
  bwrite(bp);
  brelse(bp);
}

// Blocks.

// Allocate a zeroed disk block.
// returns 0 if out of disk space.
static uint
balloc(uint dev)
{
  int b, bi, m;
  struct buf *bp;

  bp = 0;
  for(b = 0; b < sb.size; b += BPB){
    bp = bread(dev, BBLOCK(b, sb));
    for(bi = 0; bi < BPB && b + bi < sb.size; bi++){
      m = 1 << (bi % 8);
      if((bp->data[bi/8] & m) == 0){  // Is block free?
        bp->data[bi/8] |= m;  // Mark block in use.
        bwrite(bp);
        brelse(bp);
        bzero(dev, b + bi);
        return b + bi;
      }
    }
    brelse(bp);
  }
  printf("balloc: out of blocks\n");
  return 0;
}

// Free a disk block.
static void
bfree(int dev, uint b)
{
  struct buf *bp;
  int bi, m;

  bp = bread(dev, BBLOCK(b, sb));
  bi = b % BPB;
  m = 1 << (bi % 8);
  if((bp->data[bi/8] & m) == 0)
    panic("freeing free block");
  bp->data[bi/8] &= ~m;
  bwrite(bp);
  brelse(bp);
}

static uint
bmap(struct xv6fs_inode *ip, uint bn)
{
  uint addr, *a;
  struct buf *bp;

  if(bn < NDIRECT){
    if((addr = ip->addrs[bn]) == 0){
      addr = balloc(ip->dev);
      if(addr == 0)
        return 0;
      ip->addrs[bn] = addr;
    }
    return addr;
  }
  bn -= NDIRECT;

  if(bn < NINDIRECT){
    // Load indirect block, allocating if necessary.
    if((addr = ip->addrs[NDIRECT]) == 0){
      addr = balloc(ip->dev);
      if(addr == 0)
        return 0;
      ip->addrs[NDIRECT] = addr;
    }
    bp = bread(ip->dev, addr);
    a = (uint*)bp->data;
    if((addr = a[bn]) == 0){
      addr = balloc(ip->dev);
      if(addr){
        a[bn] = addr;
        bwrite(bp);
      }
    }
    brelse(bp);
    return addr;
  }

  panic("bmap: out of range");
}

struct super_block *xv6_mount(const char *source) {
  printf("mount\n");
    struct super_block *ssb = (struct super_block *)kalloc();
    if (!ssb) {
        panic("Failed to allocate super block");
        return 0;
    }

    ssb->type=&xv6fs;

    ssb->op = xv6fs.op;

    ssb->private = &sb;

    for(int i=0;i<MAXMNT;i++){
      if(root->mounts[i]==0){
        root->mounts[i]=ssb;
        break;
      }
    }


    return ssb;
}

int xv6_umount(struct super_block *sbb) {
  printf("umount\n");
    for(int i=0;i<MAXMNT;i++){
      if(root->mounts[i]==sbb){
        root->mounts[i]=0;
        kfree(sbb->private);
        kfree(sbb);
        return 0;
      }
    }
    return -1;
}

struct inode *xv6_alloc_inode(struct super_block *sbb, short type) {
  //printf("xv6_ialloc\n");
  struct xv6fs_super_block* ssb = sbb->private;
  int inum;
  struct buf *bp;
  struct dinode *dip;

  for(inum = 1; inum < ssb->ninodes; inum++){
    bp = bread(sbb->root->dev, IBLOCK(inum, (sb)));
    dip = (struct dinode*)bp->data + inum%IPB;
    if(dip->type == 0){  // a free inode
      memset(dip, 0, sizeof(*dip));
      dip->type = type;
      bwrite(bp);   // mark it allocated on the disk
      brelse(bp);
      struct inode* tt = inget(sbb->root->dev, inum);
      tt->op = &xv6fs_ops;
      tt->sb = root;
      tt->private = 0;
      return tt;
    }
    brelse(bp);
  }
  //printf("ialloc: no inodes\n");
  return 0;

}

void xv6_write_inode(struct inode *ino) {
  //printf("xv6_write_inode\n");
    struct xv6fs_inode *ip = (struct xv6fs_inode *)ino->private;
    struct buf *bp;
    struct dinode *dip;

    if (ino == 0 || ip->ref < 1) {
        panic("write_inode: invalid inode");
    }

    // 获取包含 inode 的磁盘块
    bp = bread(ino->dev, IBLOCK(ino->inum, sb));
    dip = (struct dinode*)bp->data + ino->inum % IPB;

    // 将内存中的 inode 数据写回磁盘
    dip->type = ino->type;
    dip->major = ip->major;
    dip->minor = ip->minor;
    dip->nlink = ino->nlink;
    dip->size = ino->size;
    memmove(dip->addrs, ip->addrs, sizeof(ip->addrs));

    // 将缓冲区写回磁盘
    bwrite(bp);
    brelse(bp);
}


void xv6_free_inode(struct inode *ino) {


}

void xv6_release_inode(struct inode *ino) {
  printf("xv6_release_inode\n");
  struct xv6fs_inode *de = (struct xv6fs_inode *)ino->private;
  kfree(de);
  kfree(ino);
}

int xv6_read(struct inode *ino, char dst_is_user, uint64 dst, unsigned int off, unsigned int n) {
  uint tot, m;
  struct buf *bp;
  struct xv6fs_inode *ip = (struct xv6fs_inode *)ino->private;

  if(off > ino->size || off + n < off)
    return 0;
  if(off + n > ino->size)
    n = ino->size - off;

  for(tot=0; tot<n; tot+=m, off+=m, dst+=m){
    uint addr = bmap(ip, off/BSIZE);
    if(addr == 0)
      break;
    bp = bread(ip->dev, addr);
    if(n - tot > BSIZE - off%BSIZE)
      m = BSIZE - off%BSIZE;
    else
      m = n - tot;
    if(either_copyout(dst_is_user, dst, bp->data + (off % BSIZE), m) == -1) {
      brelse(bp);
      tot = -1;
      break;
    }
    brelse(bp);
  }
  return tot;
}
int xv6_write(struct inode *ino, char src_is_user, uint64 src, unsigned int off, unsigned int n) {
  uint tot, m;
  struct buf *bp;
  struct xv6fs_inode *ip = (struct xv6fs_inode *)ino->private;

  if(off > ino->size || off + n < off)
    return -1;
  if(off + n > MAXFILE*BSIZE)
   return -1;

  for(tot=0; tot<n; tot+=m, off+=m, src+=m){
    uint addr = bmap(ip, off/BSIZE);
    if(addr == 0)
      break;
    bp = bread(ip->dev, addr);
    if(n-tot > BSIZE - off%BSIZE)
      m = BSIZE - off%BSIZE;
    else
      m = n - tot;
    if(either_copyout(src_is_user, src, bp->data + (off % BSIZE), m) == -1) {
      brelse(bp);
      tot = -1;
      break;
    }
    brelse(bp);
  }
  return tot;
}
void xv6_trunc(struct inode *ino) {
  int i, j;
  struct buf *bp;
  uint *a;
  struct xv6fs_inode *ip = (struct xv6fs_inode *)ino->private;

  for(i = 0; i < NDIRECT; i++){
    if(ip->addrs[i]){
      bfree(ip->dev, ip->addrs[i]);
      ip->addrs[i] = 0;
    }
  }

  if(ip->addrs[NDIRECT]){
    bp = bread(ip->dev, ip->addrs[NDIRECT]);
    a = (uint*)bp->data;
    for(j = 0; j < NINDIRECT; j++){
      if(a[j])
        bfree(ip->dev, a[j]);
    }
    brelse(bp);
    bfree(ip->dev, ip->addrs[NDIRECT]);
    ip->addrs[NDIRECT] = 0;
  }

  ino->size = 0;
  xv6_write_inode(ino);
}

struct file *xv6_open(struct inode *ino, unsigned int mode) {
 // printf("xv6_open\n");
  struct file *f;
  struct xv6fs_inode *ip = (struct xv6fs_inode *)ino->private;

    if(ino->type == T_DEVICE && (ip->major < 0 || ip->major >= NDEV)){
        iunlockput(ino);
        return 0;
    }

    if((f = filealloc()) == 0){
        iunlockput(ino);
        return 0;
    }

    if(ino->type == T_DEVICE){
      f->type = FD_DEVICE;
    } else {
        f->type = FD_INODE;
        f->off = 0;
    }
    f->inode = ino;
    f->readable = !(mode & O_WRONLY);
    f->writable = (mode & O_WRONLY) || (mode & O_RDWR);
    f->op = ino->op;
    if((mode & O_TRUNC) && ino->type == T_FILE){
        ino->op->trunc(ino);
    }
    return f;
}

void xv6_close(struct file *f) {
  //printf("xv6_close\n");
  struct file ff;
  if(f->ref < 1)
    panic("fileclose");
  if(--f->ref > 0){
    return;
  }
  ff = *f;
  f->ref = 0;
  f->type = FD_NONE;
  if(ff.type == FD_INODE || ff.type == FD_DEVICE)
    iput(ff.inode);
  
}

int xv6_create(struct inode *di, struct dentry *target, short type, short major, short minor) {
 // printf("xv6_create\n");
  struct inode *ip;

  ilock(di);

  struct dentry *dd = di->op->dirlookup(di, target->name);

  if((ip = dd->inode) != 0){
    iunlockput(di);
    ilock(ip);
    if(type == T_FILE && (ip->type == T_FILE || ip->type == T_DEVICE)){
        target->inode = ip;
        kfree(dd->private);
        kfree(dd);
        return 0;
    }
    iunlockput(ip);
    kfree(dd->private);
    kfree(dd);
    return 1;
  }

  if((ip = xv6_alloc_inode(root,type)) == 0){
    iunlockput(di);
    kfree(dd->private);
    kfree(dd);
    return 1;
  }
  ilock(ip);

  struct xv6fs_inode *in = ip->private;
  struct xv6fs_dentry *tar = target->private;
  strncpy(tar->name, target->name , DIRSIZ);
  
  in->major = major;
  in->minor = minor;
  ip->nlink = 1;
  ip->op->write_inode(ip);

  if(type == T_DIR){  
      if(dirlink(ip, ".", ip->inum) < 0 || dirlink(ip, "..", di->inum) < 0)
          goto fail;
  }
  if(dirlink(di, target->name, ip->inum) < 0)
      goto fail;

  if(type == T_DIR){
      di->nlink++;  
      di->op->write_inode(di);
  }

  iunlockput(di);
  target->inode = ip;
  kfree(dd->private);
  kfree(dd);
  return 0;

fail:
  ip->nlink = 0;
  ip->op->write_inode(ip);
  iunlockput(ip);
  iunlockput(di);
  kfree(dd->private);
  kfree(dd);
  return 1;
}

int xv6_link(struct dentry *target) {
  //printf("xv6_link\n");
  int off;
  uint inum = *(uint *)(target->private);
  struct inode *dp = target->parent;
  struct xv6fs_dentry de;
  struct inode *ip;
  struct dentry *dir = xv6_dirlookup(dp, target->name);
    
  if((ip = dir->inode) != 0){
      iput(ip);
      kfree(dir->private);
      kfree(dir);
      return -1;
  }

  for(off = 0; off < dp->size; off += sizeof(de)){
      if(xv6_read(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de)){
          kfree(dir->private);
          kfree(dir);
          panic("dirlink read");
      }
            
      if(de.inum == 0)
          break;
  }
  strncpy(de.name, target->name, DIRSIZ);
  de.inum = inum;
  if(xv6_write(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de)) {
      kfree(dir->private);
      kfree(dir);
      return -1;
  }

  kfree(dir->private);
  kfree(dir);
  return 0;
}

int xv6_unlink(struct dentry *d) {
  printf("xv6_unlink\n");
  struct inode *ip, *dp;
  struct xv6fs_dentry de;
  char name[DIRSIZ], path[MAXPATH];
  strncpy(path, (char *)(d->private), MAXPATH);
  uint off = 0;

    // 查找文件的 inode
    if ((dp = nameiparent(path, name)) == 0) {
        
        printf("unlink: cannot find %s\n", d->name);
        return -1;
    }

  ilock(dp);

  if(namecmp(name, ".") == 0 || namecmp(name, "..") == 0){
    iunlockput(dp);
    return -1;
  }

  struct dentry *dd = dp->op->dirlookup(dp, name);
  off = *(uint*)(dd->private);
  if((ip = dd->inode) == 0)
    goto bad;

  ilock(ip);

  if(ip->nlink < 1)
    panic("unlink: nlink < 1");
  if(ip->type == T_DIR && !xv6_isdirempty(ip)){
    iunlockput(ip);
    goto bad;
  }

  memset(&de, 0, sizeof(de));

  if(dp->op->write(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
    panic("unlink: writei");
  if(ip->type == T_DIR){
    dp->nlink--;
    dp->op->write_inode(dp);
  }
  iunlockput(dp);

  ip->nlink--;
  dp->op->write_inode(ip);
  iunlockput(ip);
  kfree(dd->private);
  kfree(dd);
  return 0;

bad:
  kfree(dd->private);
  kfree(dd);
  iunlockput(dp); 
  return -1;

}

struct dentry *xv6_dirlookup(struct inode *dir, const char *name) {
 // printf("xv6_dirlookup\n");
  uint off, inum;
  struct xv6fs_dentry dee;
  struct dentry *de = (struct dentry*)kalloc();
  de->private = (struct xv6fs_dentry*)kalloc();

  if(dir->type != T_DIR)
    panic("dirlookup not DIR");

  for(off = 0; off < dir->size; off += sizeof(dee)){
    if(xv6_read(dir, 0, (uint64)&dee, off, sizeof(dee)) != sizeof(dee))
      panic("dirlookup read");
    if(dee.inum == 0)
      continue;
    if(xv6fs_namecmp(name, dee.name) == 0){
      // entry matches path element
      inum = dee.inum;
      de->inode = inget(dir->dev, inum);
      de->op = &xv6fs_ops;
      *(uint*)(de->parent) = off;
      return de;

    }
  }
  kfree(de);
  struct dentry *ret = kalloc();
  ret->private = kalloc();
  ret->inode = 0;
  return ret;
}

void xv6_release_dentry(struct dentry *dir) {
  //printf("xv6_release_dentry\n");

    struct xv6fs_dentry *de = (struct xv6fs_dentry *)dir->private;

    kfree(de);

    kfree(dir);
}

int xv6_isdirempty(struct inode *dir) {
//  printf("xv6_isdirempty\n");
  int off;
  struct xv6fs_dentry de;

  for(off=2*sizeof(de); off<(dir->size); off+=sizeof(de)){
    if(xv6_read(dir, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
      panic("isdirempty: readi");
    if(de.inum != 0)
      return 0;
  }
  return 1;
}

void xv6_init(void) {
 // printf("xv6_init\n");
  xv6fs_fsinit(1);
}

void xv6_update_inode(struct inode *gip)  {
 // printf("xv6_update_inode\n");
  struct buf *bp;
  struct dinode *dip;
  gip->private = (struct xv6fs_inode *)kalloc();
  struct xv6fs_inode *ip = (struct xv6fs_inode *)gip->private;
//  printf("qqq\n");
  bp = bread(gip->dev, IBLOCK(gip->inum, sb));
  dip = (struct dinode*)bp->data + gip->inum%IPB;
  gip->type = dip->type;
  ip->major = dip->major;
  ip->minor = dip->minor;
  gip->nlink = dip->nlink;
  gip->size = dip->size;
  ip->dev =1;
  memmove(ip->addrs, dip->addrs, sizeof(ip->addrs));
  brelse(bp);
  if(ip->type == 0)
  panic("ilock: no type");
}


